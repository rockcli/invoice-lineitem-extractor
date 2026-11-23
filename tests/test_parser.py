from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from invoice_extractor.generator import generate_dataset
from invoice_extractor.grounding import parse_currency_cents
from invoice_extractor.models import LAYOUT_TYPES, BBoxToken
from invoice_extractor.parser import (
    _CandidateRow,
    disambiguate_candidate_rows,
    parse_invoice,
)


def test_parse_currency_cents_variants() -> None:
    assert parse_currency_cents("$1,240.00") == 124000
    assert parse_currency_cents("($45.00)") == -4500
    assert parse_currency_cents("-45.00") == -4500
    assert parse_currency_cents("-$45.00") == -4500
    assert parse_currency_cents("0.00") == 0
    assert parse_currency_cents("$0.00") == 0
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

    # Also test Case 3: over-counted definite row dropped when definite > target
    duplicate_definite = replace(definite_1, amount_cents=50000)
    resolved_3, warnings_3 = disambiguate_candidate_rows(
        [definite_1, duplicate_definite],
        target_subtotal_cents=50000,
    )
    assert len(resolved_3) == 1
    assert any("pruned" in w for w in warnings_3)
