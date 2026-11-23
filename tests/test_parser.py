from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen.canvas import Canvas

from invoice_extractor.generator import generate_dataset
from invoice_extractor.grounding import parse_currency_cents
from invoice_extractor.llm_fallback import MockFallbackProvider
from invoice_extractor.models import LAYOUT_TYPES, BBoxToken
from invoice_extractor.parser import (
    _CandidateRow,
    disambiguate_candidate_rows,
    parse_invoice,
)


def test_parse_currency_cents_variants() -> None:
    assert parse_currency_cents("$1,240.00") == 124000
    assert parse_currency_cents("$1240.00") == 124000
    assert parse_currency_cents("1240.00") == 124000
    assert parse_currency_cents("($1240.00)") == -124000
    assert parse_currency_cents("(1240.00)") == -124000
    assert parse_currency_cents("-$1240.00") == -124000
    assert parse_currency_cents("-1240.00") == -124000
    assert parse_currency_cents("($45.00)") == -4500
    assert parse_currency_cents("-45.00") == -4500
    assert parse_currency_cents("-$45.00") == -4500
    assert parse_currency_cents("0.00") == 0
    assert parse_currency_cents("$0.00") == 0
    assert parse_currency_cents("$1,24.00") is None
    assert parse_currency_cents("$12.34.56") is None
    assert parse_currency_cents("not-money") is None


@pytest.mark.parametrize("layout", LAYOUT_TYPES)
def test_parse_each_layout_accurately(tmp_path: Path, layout: str) -> None:
    dataset = generate_dataset(tmp_path, count=6, seed=42)
    gt = next(item for item in dataset if item.layout == layout)
    pdf_path = tmp_path / gt.pdf_path

    result = parse_invoice(pdf_path, layout_hint=layout)

    assert result.checksum_passed, result.checksum_detail
    assert result.invoice_no.value == gt.invoice_no
    assert result.invoice_date.value == gt.invoice_date
    assert result.subtotal_cents.value == gt.subtotal_cents
    assert result.tax_cents.value == gt.tax_cents
    assert result.credits_cents.value == gt.credits_cents
    assert result.total_cents.value == gt.total_cents
    assert len(result.line_items) == len(gt.line_items)

    for parsed_item, expected_item in zip(result.line_items, gt.line_items, strict=True):
        assert parsed_item.sku.value == expected_item.sku
        assert parsed_item.description.value == expected_item.description
        assert parsed_item.qty.value == expected_item.qty
        assert parsed_item.unit_price_cents.value == expected_item.unit_price_cents
        assert parsed_item.amount_cents.value == expected_item.amount_cents

    if layout == "multipage_table":
        assert result.pages == 2
        assert any(item.page == 1 for item in result.line_items)
        assert any(item.page == 2 for item in result.line_items)

    if layout == "group_subtotals":
        assert result.computed_line_sum_cents == gt.subtotal_cents
        assert all("subtotal" not in it.description.value.lower() for it in result.line_items)


def test_credits_discounts_negative_formats(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=12, seed=42)
    credit_gts = [g for g in dataset if g.layout == "credits_discounts"]
    assert len(credit_gts) == 2

    for gt in credit_gts:
        res = parse_invoice(tmp_path / gt.pdf_path, layout_hint=gt.layout)
        assert res.checksum_passed
        assert any(item.amount_cents.value < 0 for item in res.line_items)
        assert res.credits_cents.value == gt.credits_cents


def test_subset_sum_disambiguates_ambiguous_subtotal_vs_flat_fee() -> None:
    def make_tok(text: str, y: float) -> BBoxToken:
        return BBoxToken(page=1, x0=500.0, y0=y, x1=550.0, y1=y + 10.0, text=text)

    definite_1 = _CandidateRow(
        page=1,
        kind="definite",
        sku_tokens=[make_tok("HW-001", 100)],
        desc_tokens=[make_tok("Bearing", 100)],
        qty_tokens=[make_tok("2", 100)],
        unit_price_tokens=[make_tok("$250.00", 100)],
        amount_tokens=[make_tok("$500.00", 100)],
        qty_val=Decimal("2"),
        unit_price_cents=25000,
        amount_cents=50000,
    )
    unlabeled_group_total = _CandidateRow(
        page=1,
        kind="ambiguous",
        desc_tokens=[make_tok("Hardware", 120), make_tok("Group", 120)],
        amount_tokens=[make_tok("$500.00", 120)],
        amount_cents=50000,
    )
    flat_fee_item = _CandidateRow(
        page=1,
        kind="ambiguous",
        desc_tokens=[make_tok("Expedite", 140), make_tok("Fee", 140)],
        amount_tokens=[make_tok("$75.00", 140)],
        amount_cents=7500,
    )

    resolved, warnings = disambiguate_candidate_rows(
        [definite_1, unlabeled_group_total, flat_fee_item],
        target_subtotal_cents=57500,
    )
    assert [r.amount_cents for r in resolved] == [50000, 7500]
    assert any("subset-sum" in w for w in warnings)

    # Definite rows MUST NEVER be dropped even when definite_sum > target_subtotal_cents
    duplicate_definite = replace(definite_1, amount_cents=50000)
    resolved_3, warnings_3 = disambiguate_candidate_rows(
        [definite_1, duplicate_definite],
        target_subtotal_cents=50000,
    )
    assert len(resolved_3) == 2
    assert sum(r.amount_cents for r in resolved_3) == 100000
    assert any("preserving all 2 definite" in w for w in warnings_3)

    # Reaching max_states without an unambiguous exhaustive match must preserve definite rows
    many_ambiguous = [
        _CandidateRow(
            page=1,
            kind="ambiguous",
            desc_tokens=[make_tok(f"Opt{i}", 160 + i * 20)],
            amount_tokens=[make_tok("$10.00", 160 + i * 20)],
            amount_cents=1000,
        )
        for i in range(10)
    ]
    resolved_cap, warnings_cap = disambiguate_candidate_rows(
        [definite_1, *many_ambiguous],
        target_subtotal_cents=55000,
        max_states=5,
    )
    assert len(resolved_cap) == 1
    assert resolved_cap[0].amount_cents == 50000
    assert any("max_states=5" in w for w in warnings_cap)


def _write_custom_invoice_pdf(
    pdf_path: Path,
    *,
    rows: list[tuple[str, str, str, str, str]],
    subtotal_str: str,
    tax_str: str,
    total_str: str,
) -> None:
    c = Canvas(str(pdf_path), pagesize=LETTER)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(46, 742, "Acme Verification Supply Inc.")
    c.setFont("Helvetica", 9.5)
    c.drawString(46, 720, "Invoice No: INV-2026-9001")
    c.drawString(46, 706, "Invoice Date: 2026-11-23")

    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(50, 660, "SKU")
    c.drawString(122, 660, "Description")
    c.drawRightString(388, 660, "Qty")
    c.drawRightString(468, 660, "Unit Price")
    c.drawRightString(562, 660, "Amount")

    y = 635.0
    c.setFont("Helvetica", 8.8)
    for sku, desc, qty, unit_price, amount in rows:
        c.drawString(50, y, sku)
        c.drawString(122, y, desc)
        c.drawRightString(388, y, qty)
        c.drawRightString(468, y, unit_price)
        c.drawRightString(562, y, amount)
        y -= 22.0

    c.drawRightString(468, y - 20, "Subtotal:")
    c.drawRightString(562, y - 20, subtotal_str)
    c.drawRightString(468, y - 36, "Tax:")
    c.drawRightString(562, y - 36, tax_str)
    c.drawRightString(468, y - 54, "Grand Total:")
    c.drawRightString(562, y - 54, total_str)
    c.save()


def test_parse_invoice_supports_ungrouped_four_digit_amounts(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ungrouped_amounts.pdf"
    _write_custom_invoice_pdf(
        pdf_path,
        rows=[
            ("HW-100", "Precision Servo Drive", "1", "$1240.00", "$1240.00"),
            ("HW-200", "Linear Actuator Kit", "1", "1240.00", "1240.00"),
            ("CR-001", "Volume Rebate Credit", "1", "(1240.00)", "(1240.00)"),
        ],
        subtotal_str="1240.00",
        tax_str="0.00",
        total_str="$1240.00",
    )

    result = parse_invoice(pdf_path)
    assert result.checksum_passed, result.checksum_detail
    assert len(result.line_items) == 3
    assert [it.amount_cents.value for it in result.line_items] == [124000, 124000, -124000]
    assert result.subtotal_cents.value == 124000
    assert result.total_cents.value == 124000


def test_mismatched_invoice_subtotal_preserves_definite_items_and_triggers_fallback(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "mismatched_subtotal.pdf"
    # Two definite line items ($500.00 + $500.00 = $1,000.00), but invoice prints Subtotal $500.00.
    _write_custom_invoice_pdf(
        pdf_path,
        rows=[
            ("HW-001", "Radial Bearing Assembly", "2", "$250.00", "$500.00"),
            ("HW-002", "Stainless Flange Kit", "2", "$250.00", "$500.00"),
        ],
        subtotal_str="$500.00",
        tax_str="$0.00",
        total_str="$500.00",
    )

    spy = MockFallbackProvider(payload=None)
    result = parse_invoice(pdf_path, fallback_provider=spy)

    assert len(result.line_items) == 2
    assert [it.sku.value for it in result.line_items] == ["HW-001", "HW-002"]
    assert result.computed_line_sum_cents == 100000
    assert result.subtotal_cents.value == 50000
    assert not result.checksum_passed
    assert spy.call_count == 1
