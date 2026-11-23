from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from invoice_extractor.generator import generate_dataset
from invoice_extractor.grounding import (
    parse_currency_cents,
    tokens_row_colocated,
    verify_invoice_grounding,
    verify_line_item_arithmetic,
    verify_provenance,
)
from invoice_extractor.models import BBoxToken, Provenance
from invoice_extractor.parser import extract_tokens, parse_invoice


def test_all_layouts_have_100_percent_valid_provenance(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=6, seed=42)
    for gt in dataset:
        pdf_path = tmp_path / gt.pdf_path
        result = parse_invoice(pdf_path, layout_hint=gt.layout)
        tokens_by_page, _ = extract_tokens(pdf_path)

        assert result.all_fields_grounded()
        audit = verify_invoice_grounding(result, tokens_by_page)
        assert audit.validity_rate == 1.0, (
            f"Grounding failed for {gt.invoice_id}: {audit.failed_fields}"
        )


def test_verify_provenance_rejects_hallucinated_text_and_bad_coords(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=1, seed=42)
    pdf_path = tmp_path / dataset[0].pdf_path
    tokens_by_page, _ = extract_tokens(pdf_path)
    result = parse_invoice(pdf_path)

    valid_prov = result.total_cents.provenance
    assert valid_prov is not None
    assert verify_provenance(valid_prov, tokens_by_page)

    # Hallucinated text inside valid box
    fake_text = Provenance(
        page=valid_prov.page,
        bbox=valid_prov.bbox,
        raw_text="$999,999.99",
        source="test",
    )
    assert not verify_provenance(fake_text, tokens_by_page)

    # Non-existent page
    bad_page = Provenance(
        page=99,
        bbox=valid_prov.bbox,
        raw_text=valid_prov.raw_text,
        source="test",
    )
    assert not verify_provenance(bad_page, tokens_by_page)

    # Empty/zero-area bbox
    zero_box = Provenance(
        page=valid_prov.page,
        bbox=(10.0, 10.0, 10.0, 10.0),
        raw_text=valid_prov.raw_text,
        source="test",
    )
    assert not verify_provenance(zero_box, tokens_by_page)


def test_parse_currency_cents_ungrouped_four_plus_digits_and_malformed() -> None:
    assert parse_currency_cents("$1240.00") == 124000
    assert parse_currency_cents("1240.00") == 124000
    assert parse_currency_cents("($1240.00)") == -124000
    assert parse_currency_cents("(1240.00)") == -124000
    assert parse_currency_cents("-$1240.00") == -124000
    assert parse_currency_cents("-1240.00") == -124000
    assert parse_currency_cents("$123456.78") == 12345678
    assert parse_currency_cents("$1,240.00") == 124000

    # Malformed grouping, unbalanced parentheses, and multi-dot inputs must be rejected
    assert parse_currency_cents("$1,24.00") is None
    assert parse_currency_cents("1234,567.00") is None
    assert parse_currency_cents("$12.34.56") is None
    assert parse_currency_cents("(1240.00") is None
    assert parse_currency_cents("1240.00)") is None
    assert parse_currency_cents("$1240.5") is None


def test_line_arithmetic_and_row_colocation_helpers() -> None:
    assert verify_line_item_arithmetic(Decimal("2"), 124000, 248000)
    assert not verify_line_item_arithmetic(Decimal("2"), 124000, 248001)
    assert verify_line_item_arithmetic(Decimal("1.5"), 1001, 1502)
    assert verify_line_item_arithmetic(Decimal("1.5"), 1001, 1501)
    assert not verify_line_item_arithmetic(Decimal("1.5"), 1001, 1504)

    same_row = [
        BBoxToken(page=1, x0=50.0, y0=100.0, x1=100.0, y1=110.0, text="2"),
        BBoxToken(page=1, x0=400.0, y0=102.0, x1=450.0, y1=112.0, text="$1240.00"),
    ]
    assert tokens_row_colocated(same_row, row_tol=8.0)

    diff_row = [
        BBoxToken(page=1, x0=50.0, y0=100.0, x1=100.0, y1=110.0, text="2"),
        BBoxToken(page=1, x0=400.0, y0=130.0, x1=450.0, y1=140.0, text="$1240.00"),
    ]
    assert not tokens_row_colocated(diff_row, row_tol=8.0)

    diff_page = [
        BBoxToken(page=1, x0=50.0, y0=100.0, x1=100.0, y1=110.0, text="2"),
        BBoxToken(page=2, x0=400.0, y0=100.0, x1=450.0, y1=110.0, text="$1240.00"),
    ]
    assert not tokens_row_colocated(diff_page, row_tol=8.0)
