from __future__ import annotations

import json
from pathlib import Path

import pdfplumber

from invoice_extractor.generator import generate_dataset
from invoice_extractor.models import LAYOUT_TYPES


def test_generate_dataset_deterministic_and_balanced(tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    gt_a = generate_dataset(out_a, count=12, seed=42)
    gt_b = generate_dataset(out_b, count=12, seed=42)

    assert len(gt_a) == 12
    assert [g.to_dict() for g in gt_a] == [g.to_dict() for g in gt_b]

    layout_counts = {layout: 0 for layout in LAYOUT_TYPES}
    for gt in gt_a:
        layout_counts[gt.layout] += 1
        pdf_file = out_a / gt.pdf_path
        gt_file = out_a / "ground_truth" / f"{gt.invoice_id}.json"
        assert pdf_file.exists()
        assert gt_file.exists()

        # Ground truth arithmetic sanity check
        assert sum(item.amount_cents for item in gt.line_items) == gt.subtotal_cents
        assert gt.subtotal_cents + gt.tax_cents - gt.credits_cents == gt.total_cents

        # Verify multi-page layout genuinely produces 2 pages
        with pdfplumber.open(pdf_file) as pdf:
            if gt.layout == "multipage_table":
                assert len(pdf.pages) == 2
            else:
                assert len(pdf.pages) == 1

    assert all(count == 2 for count in layout_counts.values())

    manifest = json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["count"] == 12
    assert len(manifest["invoices"]) == 12
