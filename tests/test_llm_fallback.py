from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from invoice_extractor.generator import generate_dataset
from invoice_extractor.grounding import verify_invoice_grounding
from invoice_extractor.llm_fallback import MockFallbackProvider, apply_strict_fallback
from invoice_extractor.models import GroundedField
from invoice_extractor.parser import extract_tokens, parse_invoice


def test_fallback_never_invoked_when_checksum_passes(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=1, seed=42)
    pdf_path = tmp_path / dataset[0].pdf_path
    spy = MockFallbackProvider(payload={"unexpected": True})

    result = parse_invoice(pdf_path, fallback_provider=spy)
    assert result.checksum_passed
    assert spy.call_count == 0
    assert not result.fallback_used


def test_fallback_rejects_hallucinated_ungrounded_numbers(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=1, seed=42)
    gt = dataset[0]
    pdf_path = tmp_path / gt.pdf_path
    tokens_by_page, _ = extract_tokens(pdf_path)
    base_result = parse_invoice(pdf_path)

    broken = replace(
        base_result,
        checksum_passed=False,
        subtotal_cents=GroundedField(0, None, confidence=0.0),
    )

    hallucinated_payload = {
        "invoice_no": gt.invoice_no,
        "invoice_date": gt.invoice_date,
        "subtotal_cents": 9999999,  # Not present anywhere in PDF
        "tax_cents": gt.tax_cents,
        "credits_cents": gt.credits_cents,
        "total_cents": 9999999 + gt.tax_cents,
        "line_items": [
            {
                "sku": gt.line_items[0].sku,
                "description": gt.line_items[0].description,
                "qty": str(gt.line_items[0].qty),
                "unit_price_cents": 9999999,
                "amount_cents": 9999999,
            }
        ],
    }
    provider = MockFallbackProvider(payload=hallucinated_payload)
    repaired = apply_strict_fallback(broken, tokens_by_page, provider)

    assert provider.call_count == 1
    assert not repaired.fallback_used
    assert not repaired.checksum_passed
    assert repaired.fallback_rejected_reason is not None
    assert "ungrounded" in repaired.fallback_rejected_reason


def test_fallback_rejects_malformed_schema(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=1, seed=42)
    pdf_path = tmp_path / dataset[0].pdf_path
    tokens_by_page, _ = extract_tokens(pdf_path)
    base_result = parse_invoice(pdf_path)
    broken = replace(base_result, checksum_passed=False)

    provider = MockFallbackProvider(payload={"invoice_no": "INV-1"})
    repaired = apply_strict_fallback(broken, tokens_by_page, provider)

    assert provider.call_count == 1
    assert not repaired.fallback_used
    assert repaired.fallback_rejected_reason is not None
    assert "schema_validation_failed" in repaired.fallback_rejected_reason


def test_fallback_accepts_grounded_valid_repair_on_checksum_failure(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=1, seed=42)
    gt = dataset[0]
    pdf_path = tmp_path / gt.pdf_path
    tokens_by_page, _ = extract_tokens(pdf_path)
    base_result = parse_invoice(pdf_path)

    broken = replace(
        base_result,
        checksum_passed=False,
        line_items=[],
        checksum_detail="simulated failure",
    )

    valid_repair_payload = {
        "invoice_no": gt.invoice_no,
        "invoice_date": gt.invoice_date,
        "subtotal_cents": gt.subtotal_cents,
        "tax_cents": gt.tax_cents,
        "credits_cents": gt.credits_cents,
        "total_cents": gt.total_cents,
        "line_items": [
            {
                "sku": it.sku,
                "description": it.description,
                "qty": str(it.qty),
                "unit_price_cents": it.unit_price_cents,
                "amount_cents": it.amount_cents,
            }
            for it in gt.line_items
        ],
    }
    provider = MockFallbackProvider(payload=valid_repair_payload)
    repaired = apply_strict_fallback(broken, tokens_by_page, provider)

    assert provider.call_count == 1
    assert repaired.fallback_used
    assert repaired.checksum_passed
    assert repaired.fallback_rejected_reason is None
    assert len(repaired.line_items) == len(gt.line_items)
    assert verify_invoice_grounding(repaired, tokens_by_page).validity_rate == 1.0
