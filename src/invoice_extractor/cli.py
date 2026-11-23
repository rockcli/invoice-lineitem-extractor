"""Command-line interface for invoice generation, extraction, and evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from invoice_extractor.eval import run_evaluation
from invoice_extractor.generator import generate_dataset
from invoice_extractor.llm_fallback import DisabledFallbackProvider
from invoice_extractor.parser import parse_invoice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invoice-extractor",
        description="Deterministic geometry-first PDF invoice line-item extractor.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen_cmd = sub.add_parser("generate", help="Generate deterministic synthetic invoice PDFs.")
    gen_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Target output directory for PDFs and ground truth JSON.",
    )
    gen_cmd.add_argument("--count", type=int, default=120, help="Number of invoices to generate.")
    gen_cmd.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    ext_cmd = sub.add_parser("extract", help="Extract structured line items from a single PDF.")
    ext_cmd.add_argument("pdf", type=Path, help="Path to PDF invoice.")
    ext_cmd.add_argument(
        "--compact",
        action="store_true",
        help="Emit single-line compact JSON instead of indented JSON.",
    )

    eval_cmd = sub.add_parser(
        "eval", help="Run full benchmark evaluation and generate report/assets."
    )
    eval_cmd.add_argument("--count", type=int, default=120, help="Invoice count (default: 120).")
    eval_cmd.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    eval_cmd.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory for generated PDFs and ground truth.",
    )
    eval_cmd.add_argument(
        "--report",
        type=Path,
        default=Path("reports/eval_summary.json"),
        help="Output JSON evaluation summary path.",
    )
    eval_cmd.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Output directory for generated overview and cover PNG assets.",
    )
    eval_cmd.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip regenerating PNG visual assets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "generate":
        invoices = generate_dataset(args.output_dir, count=args.count, seed=args.seed)
        print(
            f"Generated {len(invoices)} synthetic invoices in {args.output_dir} (seed={args.seed})"
        )
        return 0

    if args.command == "extract":
        result = parse_invoice(args.pdf, fallback_provider=DisabledFallbackProvider())
        indent = None if args.compact else 2
        print(json.dumps(result.to_dict(), indent=indent))
        return 0 if result.checksum_passed else 1

    if args.command == "eval":
        summary = run_evaluation(
            data_dir=args.data_dir,
            report_path=args.report,
            assets_dir=args.assets_dir,
            count=args.count,
            seed=args.seed,
            generate_assets=not args.skip_assets,
        )
        print(f"Evaluated {summary.total_invoices} invoices (seed={summary.seed}) -> {args.report}")
        print(
            f"  Line-Item P/R/F1 : {summary.line_item_precision:.4f} / "
            f"{summary.line_item_recall:.4f} / {summary.line_item_f1:.4f}"
        )
        print(
            f"  Header    P/R/F1 : {summary.header_precision:.4f} / "
            f"{summary.header_recall:.4f} / {summary.header_f1:.4f}"
        )
        print(
            f"  Overall   P/R/F1 : {summary.overall_field_precision:.4f} / "
            f"{summary.overall_field_recall:.4f} / {summary.overall_field_f1:.4f}"
        )
        print(
            f"  Exact Total Pass : {summary.exact_total_pass_count}/{summary.total_invoices} "
            f"({summary.checksum_pass_rate * 100:.2f}%)"
        )
        print(f"  BBox Provenance  : {summary.provenance_validity_rate * 100:.2f}% valid")
        print(
            f"  Latency p50 / p95: {summary.p50_latency_ms:.2f} ms / "
            f"{summary.p95_latency_ms:.2f} ms"
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
