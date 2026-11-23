# Invoice Line-Item Extractor

**Reference implementation · synthetic data**

Automated AP and procurement pipelines break when multi-line invoice descriptions, group subtotals, or OCR coordinate jitter cause line-item sums to drift from the invoice grand total without an auditable trail back to the PDF.

![Grounded Extraction & Benchmark](assets/demo_overview.png)

## Quick start

```bash
uv sync && make eval
make test
```

## Reproducible benchmark (`120` synthetic PDFs, `seed=42`)

Metrics below are produced directly by `make eval` (`uv run invoice-extractor eval --count 120 --seed 42` -> `reports/eval_summary.json`):

| Metric | Result |
| :--- | :--- |
| Dataset | `120` synthetic PDFs across `6` vendor layouts (`20` per layout, `seed=42`) |
| Line-Item Precision / Recall / F1 | `1.0000` / `1.0000` / `1.0000` |
| Header Field Precision / Recall / F1 | `1.0000` / `1.0000` / `1.0000` |
| Overall Field Precision / Recall / F1 | `1.0000` / `1.0000` / `1.0000` |
| Exact Grand-Total Checksum Pass | `120 / 120` (`100.00%`) |
| Token BBox Provenance Validity | `100.00%` |
| Per-Invoice Latency (`p50` / `p95` / `mean`) | `16.83 ms` / `33.11 ms` / `19.45 ms` |

### Per-layout breakdown

| Layout | Count | Field F1 | Exact Total Pass | BBox Valid | p50 (ms) | p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `standard_grid` | 20 | `1.0000` | `100.0%` | `100.0%` | `14.15` | `17.30` |
| `wrapped_header` | 20 | `1.0000` | `100.0%` | `100.0%` | `16.04` | `38.53` |
| `multipage_table` | 20 | `1.0000` | `100.0%` | `100.0%` | `30.20` | `49.03` |
| `credits_discounts` | 20 | `1.0000` | `100.0%` | `100.0%` | `15.96` | `30.91` |
| `group_subtotals` | 20 | `1.0000` | `100.0%` | `100.0%` | `18.75` | `20.32` |
| `jittered_ocr_layout` | 20 | `1.0000` | `100.0%` | `100.0%` | `15.42` | `17.77` |

## What breaks in production & how this handles it

1. **2-line split column headers** (`QTY / UNIT`, `EXT'D / NET`) — [`_detect_header_in_rows()`](src/invoice_extractor/parser.py) and [`_cluster_header_columns()`](src/invoice_extractor/parser.py) merge vertically adjacent header rows (`<= 10pt` gap) and cluster tokens horizontally before inferring right-edge numeric anchors via median `x1` clustering.
2. **Wrapped descriptions without qty/price columns** — [`_parse_table_rows()`](src/invoice_extractor/parser.py) stitches description-column continuation rows onto the active line item, resets active state at section headers or subtotals, and builds a union bounding box via [`build_provenance()`](src/invoice_extractor/grounding.py).
3. **Group subtotals double-counting line sums** — [`_parse_table_rows()`](src/invoice_extractor/parser.py) isolates labeled subtotal rows and ambiguous amount-only rows, then [`disambiguate_candidate_rows()`](src/invoice_extractor/parser.py) and [`find_exact_subset_indices()`](src/invoice_extractor/parser.py) resolve exact-cent subsets so intermediate section subtotals never inflate `sum(line_items)`.
4. **Floating-point cent rounding drift** — [`parse_currency_cents()`](src/invoice_extractor/grounding.py) and [`parse_decimal_qty()`](src/invoice_extractor/grounding.py) convert `($45.00)`, `-45.00`, and fractional quantities into exact `int` cents and `Decimal` quantities before reconciling `sum(line_items) == subtotal` and `subtotal + tax - credits == total`.
5. **Ungrounded LLM hallucinations** — [`apply_strict_fallback()`](src/invoice_extractor/llm_fallback.py) enforces a strict checksum gate (`checksum_passed == False` only), validates the JSON schema, and verifies every proposed number and SKU against actual PDF tokens via [`find_cents_token()`](src/invoice_extractor/grounding.py), [`find_qty_token()`](src/invoice_extractor/grounding.py), and [`find_text_token_span()`](src/invoice_extractor/grounding.py), rejecting any ungrounded value.

## Limitations

- Designed for born-digital PDFs or OCR-word-boxed PDFs that carry token bounding-box coordinates; raw raster images or un-OCRed scans require an upstream OCR box provider (such as Tesseract, PaddleOCR, or DocumentAI) before calling the geometry engine.
- Evaluation covers `6` deterministic synthetic vendor templates (`120` PDFs); unseen production vendor layouts with rotated tables, merged multi-currency columns, or handwritten annotations may fail the deterministic checksum and trigger the fallback gate.
- Ships with offline `disabled` and `mock` fallback providers (`INVOICE_EXTRACTOR_FALLBACK_MODE=disabled`) so zero external API keys or network calls are required.

## CLI commands

```bash
uv run invoice-extractor generate --count 120 --seed 42
uv run invoice-extractor extract data/generated/pdfs/001_standard_grid.pdf
uv run invoice-extractor eval --count 120 --seed 42
```

Built with AI-assisted tooling; architecture, review, and tests by me.
