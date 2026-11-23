# Design Notes & Architecture Walkthrough

**Reference implementation · synthetic data**

## 1. Why Geometry + Exact-Cent Checksum Beats LLM-First Extraction

Sending every invoice PDF page to a multimodal LLM introduces three production liabilities: latency (seconds per page vs. `16.83 ms` p50 locally), silent numeric drift (floating-point rounding or plausible-looking hallucinations when dense tables wrap), and ungrounded outputs that finance teams cannot click back to a source coordinate.

This reference implementation inverts the control flow:

1. **Geometry first (`parser.py`)**: Extract word tokens `(page, x0, y0, x1, y1, text)`, cluster jitter-tolerant horizontal lines (`±2.5pt` tolerance via running cluster mean), dynamically detect 1-line or 2-line column header bands, and refine right-aligned numeric anchors (`Qty`, `Unit Price`, `Amount`) from body-token `x1` medians.
2. **Exact-cent gate (`int` cents + `Decimal` qty)**: Every monetary string (`$1,240.00`, `($45.00)`, `-45.00`) parses strictly through `Decimal` string scaling into signed `int` cents. Acceptance requires both invariants:
   - `sum(item.amount_cents for item in line_items) == subtotal_cents`
   - `subtotal_cents + tax_cents - credits_cents == total_cents`
3. **Strict-gated fallback (`llm_fallback.py`)**: The fallback interface is structurally unreachable when `checksum_passed == True`. Only invoices failing arithmetic reconciliation can invoke a `FallbackProvider`, and any proposed repair is rejected unless its JSON passes schema validation, every proposed SKU/quantity/amount grounds to real PDF tokens, and the repaired values reconcile to the exact cent.

## 2. Subset-Sum Subtotal Disambiguation Tradeoff

Vendor invoices frequently embed intermediate category totals (`Section A Subtotal`, `Group C Package Total`) directly inside the `Amount` column. Label heuristics catch standard prefixes, yet real vendors sometimes omit `"Subtotal"` on category rollups while simultaneously placing legitimate flat-fee line items (`Expedite Fee $75.00`) that lack `Qty` and `Unit Price`.

`disambiguate_candidate_rows()` handles both:

- Rows with verified `qty * unit_price_cents == amount_cents` and non-subtotal labels form the `definite` set.
- If `sum(definite) == subtotal_cents`, all labeled subtotals and amount-only summary rows are safely discarded.
- If `sum(definite) < subtotal_cents`, `find_exact_subset_indices()` runs bounded branch-and-bound subset-sum over `ambiguous` rows targeting `subtotal_cents - sum(definite)`, preferring higher-cardinality subsets so genuine flat-fee charges stay while group rollups drop.
- If `sum(definite) > subtotal_cents`, subset-sum runs over the non-subtotal candidate pool to prune accidental double-counts.

**Tradeoff**: Bounded branch-and-bound (`max_states=50_000`) is exact and sub-millisecond for realistic invoices (`< 60` rows). Its theoretical edge case is a coincidence where two unrelated ambiguous rows equal the exact same cent residual as a genuine item; in production, adding a positional check (whether a candidate sits directly beneath a section break) breaks ties deterministically.

## 3. Token-BBox Grounding Contract

Every extracted header field (`invoice_no`, `invoice_date`, `subtotal_cents`, `tax_cents`, `credits_cents` when non-zero, `total_cents`) and every `LineItem` field (`sku`, `description`, `qty`, `unit_price_cents`, `amount_cents`) is wrapped in `GroundedField[T]` with `Provenance(page, bbox, raw_text, source)`.

Key invariants enforced by `grounding.py`:

- **Reading-order stability under OCR jitter**: `cluster_tokens_into_rows()` groups tokens within `5.2pt` vertical tolerance before sorting left-to-right by `x0`. `build_provenance()` uses this same reading-order clustering so `±2.5pt` vertical token jitter never reverses adjacent words inside `raw_text`.
- **Verification contract (`verify_provenance`)**: Every provenance must have `page >= 1`, finite non-degenerate box coordinates (`x1 > x0`, `y1 > y0`), intersecting PDF tokens on that page within `1.0pt` tolerance, and normalized token text matching `raw_text`.
- **Anti-hallucination enforcement**: `apply_strict_fallback()` uses `find_text_token_span()`, `find_cents_token()`, and `find_qty_token()` to prove every fallback-proposed field exists verbatim on the PDF before accepting the repair.

## 4. Five-Minute Walkthrough Talking Points

- **0:00–1:00 — Problem & Invariant**: Why naive LLM extraction fails AP auditability and how exact-cent accounting equations turn PDF parsing into a self-verifying pipeline.
- **1:00–2:15 — Geometry Pipeline**: Word bbox extraction -> jitter-tolerant row clustering -> 2-line header band detection -> `x1` median anchor refinement -> wrapped description stitching & cross-page continuation.
- **2:15–3:15 — Group Subtotals & Negative Credits**: Handling `($45.00)` / `-45.00` accounting formats, preventing double-counting of intermediate section subtotals, and bounded subset-sum disambiguation.
- **3:15–4:15 — Provenance & Strict Fallback Gate**: How `GroundedField[T]` traces every dollar back to `(page, x0, y0, x1, y1)` and how the fallback gate rejects hallucinated numbers in offline tests.
- **4:15–5:00 — Benchmark & Production Path**: `120` synthetic invoices across `6` layouts (`1.0000` F1, `120/120` exact total pass, `16.83 ms` p50), plus plugging in an upstream OCR token provider (Tesseract/PaddleOCR/DocumentAI) for scanned image PDFs.
