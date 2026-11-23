from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from invoice_extractor.generator import generate_dataset
from invoice_extractor.grounding import verify_invoice_grounding
from invoice_extractor.llm_fallback import (
    DisabledFallbackProvider,
    MockFallbackProvider,
    apply_strict_fallback,
    resolve_fallback_provider,
)
from invoice_extractor.models import BBoxToken, GroundedField
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


def test_fallback_accepts_wrapped_multiline_description_on_same_row_anchor(
    tmp_path: Path,
) -> None:
    dataset = generate_dataset(tmp_path, count=2, seed=42)
    gt = next(item for item in dataset if item.layout == "wrapped_header")
    pdf_path = tmp_path / gt.pdf_path
    tokens_by_page, _ = extract_tokens(pdf_path)
    base_result = parse_invoice(pdf_path)

    broken = replace(base_result, checksum_passed=False, line_items=[])
    valid_payload = {
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
    provider = MockFallbackProvider(payload=valid_payload)
    repaired = apply_strict_fallback(broken, tokens_by_page, provider)
    assert repaired.fallback_used, repaired.fallback_rejected_reason
    assert repaired.checksum_passed
    assert verify_invoice_grounding(repaired, tokens_by_page).validity_rate == 1.0


def _synthetic_two_row_tokens() -> dict[int, list[BBoxToken]]:
    def tok(x0: float, y0: float, x1: float, text: str) -> BBoxToken:
        return BBoxToken(page=1, x0=x0, y0=y0, x1=x1, y1=y0 + 10.0, text=text)

    return {
        1: [
            tok(46, 80, 120, "INV-2026-8801"),
            tok(46, 96, 110, "2026-11-23"),
            # Row 1 (y=200): 2 * $150.00 = $300.00
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 390, "2"),
            tok(440, 200, 485, "$150.00"),
            tok(520, 200, 565, "$300.00"),
            # Row 2 (y=240): 3 * $100.00 = $300.00
            tok(50, 240, 95, "SKU-B"),
            tok(120, 240, 165, "Gadget"),
            tok(380, 240, 390, "3"),
            tok(440, 240, 485, "$100.00"),
            tok(520, 240, 565, "$300.00"),
            # Summary (y=320..360): Subtotal $300.00, Tax $0.00, Total $300.00 (and $600.00 extra)
            tok(520, 320, 565, "$300.00"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$300.00"),
            tok(520, 380, 565, "$600.00"),
        ]
    }


def _dummy_broken_result() -> object:
    from invoice_extractor.models import InvoiceResult

    return InvoiceResult(
        invoice_no=GroundedField("INV-2026-8801", None, confidence=0.0),
        invoice_date=GroundedField("2026-11-23", None, confidence=0.0),
        subtotal_cents=GroundedField(0, None, confidence=0.0),
        tax_cents=GroundedField(0, None, confidence=0.0),
        credits_cents=GroundedField(0, None, confidence=0.0),
        total_cents=GroundedField(0, None, confidence=0.0),
        line_items=[],
        checksum_passed=False,
        checksum_detail="simulated failure",
    )


def test_fallback_rejects_cross_row_frankenstein_combination() -> None:
    tokens_by_page = _synthetic_two_row_tokens()
    broken = _dummy_broken_result()
    # Takes SKU-A + Widget + $150.00 from Row 1 (y=200), qty=3 from Row 2 (y=240),
    # or qty=3 + $100.00 from Row 2 paired with SKU-A from Row 1 -> cross-row assembly!
    cross_row_payload = {
        "invoice_no": "INV-2026-8801",
        "invoice_date": "2026-11-23",
        "subtotal_cents": 30000,
        "tax_cents": 0,
        "credits_cents": 0,
        "total_cents": 30000,
        "line_items": [
            {
                "sku": "SKU-A",  # Row 1 (y=200)
                "description": "Widget",  # Row 1 (y=200)
                "qty": "3",  # Row 2 (y=240)
                "unit_price_cents": 10000,  # Row 2 (y=240)
                "amount_cents": 30000,  # Row 1 or Row 2
            }
        ],
    }
    provider = MockFallbackProvider(payload=cross_row_payload)
    repaired = apply_strict_fallback(broken, tokens_by_page, provider)  # type: ignore[arg-type]

    assert not repaired.fallback_used
    assert not repaired.checksum_passed
    assert repaired.fallback_rejected_reason is not None
    assert "ungrounded" in repaired.fallback_rejected_reason


def test_fallback_rejects_per_line_arithmetic_mismatch() -> None:
    def tok(x0: float, y0: float, x1: float, text: str) -> BBoxToken:
        return BBoxToken(page=1, x0=x0, y0=y0, x1=x1, y1=y0 + 10.0, text=text)

    # All tokens on Row 1 (y=200) and in Summary (y=320..360) are genuinely present & co-located,
    # except Row 1 has a line-item arithmetic error: 2 * $150.00 != $600.00.
    bad_math_tokens = {
        1: [
            tok(46, 80, 120, "INV-2026-8803"),
            tok(46, 96, 110, "2026-11-23"),
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 390, "2"),
            tok(440, 200, 485, "$150.00"),
            tok(520, 200, 565, "$600.00"),
            tok(520, 320, 565, "$600.00"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$600.00"),
        ]
    }
    broken = _dummy_broken_result()
    bad_math_payload = {
        "invoice_no": "INV-2026-8803",
        "invoice_date": "2026-11-23",
        "subtotal_cents": 60000,
        "tax_cents": 0,
        "credits_cents": 0,
        "total_cents": 60000,
        "line_items": [
            {
                "sku": "SKU-A",
                "description": "Widget",
                "qty": "2",
                "unit_price_cents": 15000,
                "amount_cents": 60000,
            }
        ],
    }
    provider = MockFallbackProvider(payload=bad_math_payload)
    repaired = apply_strict_fallback(broken, bad_math_tokens, provider)  # type: ignore[arg-type]

    assert not repaired.fallback_used
    assert not repaired.checksum_passed
    assert repaired.fallback_rejected_reason is not None
    assert "line_arithmetic_failed" in repaired.fallback_rejected_reason


def test_fallback_rejects_integer_one_cent_drift_and_allows_fractional_rounding() -> None:
    def tok(x0: float, y0: float, x1: float, text: str) -> BBoxToken:
        return BBoxToken(page=1, x0=x0, y0=y0, x1=x1, y1=y0 + 10.0, text=text)

    # Integer qty=2 * $10.00 = $20.00 -> $20.01 must be rejected (0 cent tolerance for int qty).
    int_drift_tokens = {
        1: [
            tok(46, 80, 120, "INV-2026-8804"),
            tok(46, 96, 110, "2026-11-23"),
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 390, "2"),
            tok(440, 200, 485, "$10.00"),
            tok(520, 200, 565, "$20.01"),
            tok(520, 320, 565, "$20.01"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$20.01"),
        ]
    }
    broken = _dummy_broken_result()
    int_drift_payload = {
        "invoice_no": "INV-2026-8804",
        "invoice_date": "2026-11-23",
        "subtotal_cents": 2001,
        "tax_cents": 0,
        "credits_cents": 0,
        "total_cents": 2001,
        "line_items": [
            {
                "sku": "SKU-A",
                "description": "Widget",
                "qty": "2",
                "unit_price_cents": 1000,
                "amount_cents": 2001,
            }
        ],
    }
    repaired = apply_strict_fallback(
        broken,  # type: ignore[arg-type]
        int_drift_tokens,
        MockFallbackProvider(payload=int_drift_payload),
    )
    assert not repaired.fallback_used
    assert repaired.fallback_rejected_reason is not None
    assert "line_arithmetic_failed" in repaired.fallback_rejected_reason

    # Fractional qty=1.5 * $10.01 = 1501.5 cents -> 1502 cents passes (<=1c fractional tolerance).
    frac_tokens = {
        1: [
            tok(46, 80, 120, "INV-2026-8805"),
            tok(46, 96, 110, "2026-11-23"),
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 395, "1.5"),
            tok(440, 200, 485, "$10.01"),
            tok(520, 200, 565, "$15.02"),
            tok(520, 320, 565, "$15.02"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$15.02"),
        ]
    }
    frac_payload = {
        "invoice_no": "INV-2026-8805",
        "invoice_date": "2026-11-23",
        "subtotal_cents": 1502,
        "tax_cents": 0,
        "credits_cents": 0,
        "total_cents": 1502,
        "line_items": [
            {
                "sku": "SKU-A",
                "description": "Widget",
                "qty": "1.5",
                "unit_price_cents": 1001,
                "amount_cents": 1502,
            }
        ],
    }
    frac_ok = apply_strict_fallback(
        broken,  # type: ignore[arg-type]
        frac_tokens,
        MockFallbackProvider(payload=frac_payload),
    )
    assert frac_ok.fallback_used, frac_ok.fallback_rejected_reason
    assert frac_ok.checksum_passed


def test_fallback_rejects_reusing_same_amount_token_across_two_lines() -> None:
    # Only ONE line row exists on the PDF ($300.00), while Subtotal & Total show $600.00.
    # Fallback tries to claim TWO lines referencing the same Row 1 tokens ($300.00 + $300.00).
    def tok(x0: float, y0: float, x1: float, text: str) -> BBoxToken:
        return BBoxToken(page=1, x0=x0, y0=y0, x1=x1, y1=y0 + 10.0, text=text)

    single_row_tokens = {
        1: [
            tok(46, 80, 120, "INV-2026-8802"),
            tok(46, 96, 110, "2026-11-23"),
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 390, "2"),
            tok(440, 200, 485, "$150.00"),
            tok(520, 200, 565, "$300.00"),
            tok(520, 320, 565, "$600.00"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$600.00"),
        ]
    }
    broken = _dummy_broken_result()
    duplicate_claim_payload = {
        "invoice_no": "INV-2026-8802",
        "invoice_date": "2026-11-23",
        "subtotal_cents": 60000,
        "tax_cents": 0,
        "credits_cents": 0,
        "total_cents": 60000,
        "line_items": [
            {
                "sku": "SKU-A",
                "description": "Widget",
                "qty": "2",
                "unit_price_cents": 15000,
                "amount_cents": 30000,
            },
            {
                "sku": "SKU-A",
                "description": "Widget",
                "qty": "2",
                "unit_price_cents": 15000,
                "amount_cents": 30000,
            },
        ],
    }
    provider = MockFallbackProvider(payload=duplicate_claim_payload)
    repaired = apply_strict_fallback(broken, single_row_tokens, provider)  # type: ignore[arg-type]

    assert not repaired.fallback_used
    assert not repaired.checksum_passed
    assert repaired.fallback_rejected_reason is not None
    assert "ungrounded" in repaired.fallback_rejected_reason


def test_resolve_fallback_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVOICE_EXTRACTOR_FALLBACK_MODE", raising=False)
    assert isinstance(resolve_fallback_provider(), DisabledFallbackProvider)

    monkeypatch.setenv("INVOICE_EXTRACTOR_FALLBACK_MODE", "mock")
    assert isinstance(resolve_fallback_provider(), MockFallbackProvider)

    monkeypatch.setenv("INVOICE_EXTRACTOR_FALLBACK_MODE", "disabled")
    assert isinstance(resolve_fallback_provider(), DisabledFallbackProvider)

    monkeypatch.setenv("INVOICE_EXTRACTOR_FALLBACK_MODE", "invalid_mode")
    with pytest.raises(ValueError, match="Unsupported INVOICE_EXTRACTOR_FALLBACK_MODE"):
        resolve_fallback_provider()


def test_fallback_rejects_cross_row_multiline_description_swallowing() -> None:
    def tok(x0: float, y0: float, x1: float, text: str) -> BBoxToken:
        return BBoxToken(page=1, x0=x0, y0=y0, x1=x1, y1=y0 + 10.0, text=text)

    # Row 1 (y=200) and Row 2 (y=214) are 14pt apart (< max_wrap_dy=18pt).
    # A hallucinated proposal merges Row 1's description with Row 2's SKU/description
    # (or the entire Row 2), which must be rejected.
    tight_rows_tokens = {
        1: [
            tok(46, 80, 120, "INV-2026-8806"),
            tok(46, 96, 110, "2026-11-23"),
            # Row 1 (y=200, cy=205)
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 390, "2"),
            tok(440, 200, 485, "$150.00"),
            tok(520, 200, 565, "$300.00"),
            # Row 2 (y=214, cy=219 -> 14pt below Row 1)
            tok(50, 214, 95, "SKU-B"),
            tok(120, 214, 165, "Gadget"),
            tok(380, 214, 390, "3"),
            tok(440, 214, 485, "$100.00"),
            tok(520, 214, 565, "$300.00"),
            # Summary
            tok(520, 320, 565, "$300.00"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$300.00"),
        ]
    }
    broken = _dummy_broken_result()

    for swallowed_desc in (
        "Widget SKU-B Gadget",
        "Widget SKU-B",
        "Widget SKU-B Gadget 3 $100.00 $300.00",
    ):
        swallowing_payload = {
            "invoice_no": "INV-2026-8806",
            "invoice_date": "2026-11-23",
            "subtotal_cents": 30000,
            "tax_cents": 0,
            "credits_cents": 0,
            "total_cents": 30000,
            "line_items": [
                {
                    "sku": "SKU-A",
                    "description": swallowed_desc,
                    "qty": "2",
                    "unit_price_cents": 15000,
                    "amount_cents": 30000,
                }
            ],
        }
        repaired = apply_strict_fallback(
            broken,  # type: ignore[arg-type]
            tight_rows_tokens,
            MockFallbackProvider(payload=swallowing_payload),
        )
        assert not repaired.fallback_used, f"Unexpectedly accepted swallowed desc: {swallowed_desc}"
        assert not repaired.checksum_passed
        assert repaired.fallback_rejected_reason is not None
        assert "ungrounded" in repaired.fallback_rejected_reason

    # Even if Row 2 omits an SKU token before "Gadget", jumping 14pt into Row 2
    # (which carries qty/unit/amount row anchors) must still be rejected.
    no_sku_row2_tokens = {
        1: [
            tok(46, 80, 120, "INV-2026-8806"),
            tok(46, 96, 110, "2026-11-23"),
            tok(50, 200, 95, "SKU-A"),
            tok(120, 200, 165, "Widget"),
            tok(380, 200, 390, "2"),
            tok(440, 200, 485, "$150.00"),
            tok(520, 200, 565, "$300.00"),
            tok(120, 214, 165, "Gadget"),
            tok(380, 214, 390, "3"),
            tok(440, 214, 485, "$100.00"),
            tok(520, 214, 565, "$300.00"),
            tok(520, 320, 565, "$300.00"),
            tok(520, 340, 555, "$0.00"),
            tok(520, 360, 565, "$300.00"),
        ]
    }
    repaired_no_sku = apply_strict_fallback(
        broken,  # type: ignore[arg-type]
        no_sku_row2_tokens,
        MockFallbackProvider(
            payload={
                "invoice_no": "INV-2026-8806",
                "invoice_date": "2026-11-23",
                "subtotal_cents": 30000,
                "tax_cents": 0,
                "credits_cents": 0,
                "total_cents": 30000,
                "line_items": [
                    {
                        "sku": "SKU-A",
                        "description": "Widget Gadget",
                        "qty": "2",
                        "unit_price_cents": 15000,
                        "amount_cents": 30000,
                    }
                ],
            }
        ),
    )
    assert not repaired_no_sku.fallback_used
    assert not repaired_no_sku.checksum_passed
    assert repaired_no_sku.fallback_rejected_reason is not None
    assert "ungrounded" in repaired_no_sku.fallback_rejected_reason


def test_fallback_accepts_jittered_ocr_wrapped_description(tmp_path: Path) -> None:
    dataset = generate_dataset(tmp_path, count=12, seed=42)
    jittered_items = [item for item in dataset if item.layout == "jittered_ocr_layout"]
    assert jittered_items
    for gt in jittered_items:
        pdf_path = tmp_path / gt.pdf_path
        tokens_by_page, _ = extract_tokens(pdf_path)
        base_result = parse_invoice(pdf_path)
        broken = replace(base_result, checksum_passed=False, line_items=[])
        valid_payload = {
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
        repaired = apply_strict_fallback(
            broken, tokens_by_page, MockFallbackProvider(payload=valid_payload)
        )
        assert repaired.fallback_used, repaired.fallback_rejected_reason
        assert repaired.checksum_passed
        assert verify_invoice_grounding(repaired, tokens_by_page).validity_rate == 1.0
