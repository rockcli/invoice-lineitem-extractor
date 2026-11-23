from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from invoice_extractor.cli import main


def test_cli_generate_extract_and_eval(tmp_path: Path, capsys: object) -> None:
    data_dir = tmp_path / "generated"
    report_path = tmp_path / "reports" / "eval_summary.json"
    assets_dir = tmp_path / "assets"

    assert main(["generate", "--output-dir", str(data_dir), "--count", "6", "--seed", "42"]) == 0
    sample_pdf = data_dir / "pdfs" / "001_standard_grid.pdf"
    assert sample_pdf.exists()

    assert main(["extract", str(sample_pdf), "--compact"]) == 0
    assert (
        main(
            [
                "eval",
                "--count",
                "12",
                "--seed",
                "42",
                "--data-dir",
                str(data_dir),
                "--report",
                str(report_path),
                "--assets-dir",
                str(assets_dir),
            ]
        )
        == 0
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["total_invoices"] == 12
    assert payload["exact_total_pass_count"] == 12
    assert payload["overall_field_f1"] == 1.0
    assert payload["provenance_validity_rate"] == 1.0

    cover_path = assets_dir / "cover_1600x1200.png"
    overview_path = assets_dir / "demo_overview.png"
    assert cover_path.exists()
    assert overview_path.exists()
    with Image.open(cover_path) as im:
        assert im.size == (1600, 1200)


def test_cli_extract_respects_fallback_mode_env_and_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "generated"
    assert main(["generate", "--output-dir", str(data_dir), "--count", "1", "--seed", "42"]) == 0
    sample_pdf = data_dir / "pdfs" / "001_standard_grid.pdf"

    monkeypatch.setenv("INVOICE_EXTRACTOR_FALLBACK_MODE", "mock")
    assert main(["extract", str(sample_pdf), "--compact"]) == 0
    assert main(["extract", str(sample_pdf), "--compact", "--fallback-mode", "disabled"]) == 0
