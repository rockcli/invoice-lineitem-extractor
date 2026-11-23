from __future__ import annotations

from pathlib import Path

from invoice_extractor.generator import generate_dataset
from invoice_extractor.grounding import verify_invoice_grounding, verify_provenance
from invoice_extractor.models import Provenance
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
