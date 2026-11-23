"""Deterministic geometry-first PDF invoice parser and exact-cent reconciler."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from statistics import median

import pdfplumber

from invoice_extractor.grounding import (
    build_provenance,
    cluster_tokens_into_rows,
    normalize_text,
    parse_currency_cents,
    parse_decimal_qty,
)
from invoice_extractor.llm_fallback import FallbackProvider, apply_strict_fallback
from invoice_extractor.models import (
    BBoxToken,
    GroundedField,
    InvoiceResult,
    LineItem,
    Provenance,
    union_bbox,
)

_SKU_HDR_WORDS = {"sku", "item", "code", "part"}
_DESC_HDR_WORDS = {"description", "product", "service", "details"}
_QTY_HDR_WORDS = {"qty", "qty/", "quantity", "units", "order"}
_UNIT_PRICE_HDR_WORDS = {"price", "rate"}
_AMOUNT_HDR_WORDS = {"amount", "ext'd", "extd", "extended", "net", "total"}

_FOOTER_NOISE_RE = re.compile(
    r"(?:^page\s+\d+\s+of\s+\d+|^continued\s+on\s+page|^ocr-conf:|reference\s+synthetic\s+copy)",
    re.IGNORECASE,
)
_SUBTOTAL_LABEL_RE = re.compile(
    r"\b(?:sub-?total|group\s+[a-z0-9]+\s+.*total|package\s+total|summary\s+total|section\s+[a-z0-9]+\s+summary)\b",
    re.IGNORECASE,
)
_SECTION_HEADER_RE = re.compile(r"^(?:section|group)\s+[a-z0-9]+[:\s-]", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$")
_INV_NO_RE = re.compile(r"^INV-[A-Z0-9-]+$", re.IGNORECASE)


@dataclass(frozen=True)
class TableColumns:
    """Inferred horizontal boundaries for a 5-column invoice table."""

    header_end_y: float
    sku_x0: float
    desc_x0: float
    sku_desc_split: float
    desc_end_x: float
    qty_anchor_x1: float
    unit_price_anchor_x1: float
    amount_anchor_x1: float


@dataclass
class _CandidateRow:
    page: int
    kind: str  # "definite", "subtotal_label", "ambiguous"
    sku_tokens: list[BBoxToken] = field(default_factory=list)
    desc_tokens: list[BBoxToken] = field(default_factory=list)
    qty_tokens: list[BBoxToken] = field(default_factory=list)
    unit_price_tokens: list[BBoxToken] = field(default_factory=list)
    amount_tokens: list[BBoxToken] = field(default_factory=list)
    qty_val: Decimal = Decimal("1")
    unit_price_cents: int = 0
    amount_cents: int = 0

    def to_line_item(self, *, source: str) -> LineItem:
        all_row_toks = [
            *self.sku_tokens,
            *self.desc_tokens,
            *self.qty_tokens,
            *self.unit_price_tokens,
            *self.amount_tokens,
        ]
        row_bbox = union_bbox([t.bbox for t in all_row_toks]) if all_row_toks else None
        sku_text = normalize_text(" ".join(t.text for t in self.sku_tokens)) or "FLAT-FEE"
        desc_text = normalize_text(" ".join(t.text for t in self.desc_tokens))
        sku_prov = (
            build_provenance(self.sku_tokens, source=source)
            if self.sku_tokens
            else build_provenance(self.desc_tokens[:1], source=source)
        )
        desc_prov = build_provenance(self.desc_tokens, source=source)
        qty_prov = (
            build_provenance(self.qty_tokens, source=source)
            if self.qty_tokens
            else build_provenance(self.amount_tokens, source=source)
        )
        unit_prov = (
            build_provenance(self.unit_price_tokens, source=source)
            if self.unit_price_tokens
            else build_provenance(self.amount_tokens, source=source)
        )
        amt_prov = build_provenance(self.amount_tokens, source=source)

        return LineItem(
            sku=GroundedField(sku_text, sku_prov, confidence=1.0),
            description=GroundedField(desc_text, desc_prov, confidence=1.0),
            qty=GroundedField(self.qty_val, qty_prov, confidence=1.0),
            unit_price_cents=GroundedField(self.unit_price_cents, unit_prov, confidence=1.0),
            amount_cents=GroundedField(self.amount_cents, amt_prov, confidence=1.0),
            page=self.page,
            row_bbox=row_bbox,
            is_credit=self.amount_cents < 0,
        )


def extract_tokens(pdf_path: Path) -> tuple[dict[int, list[BBoxToken]], int]:
    """Extract per-page word bounding boxes using pdfplumber."""
    tokens_by_page: dict[int, list[BBoxToken]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            raw_words = page.extract_words(
                x_tolerance=1.0,
                y_tolerance=3.0,
                keep_blank_chars=False,
                use_text_flow=False,
            )
            page_tokens = [
                BBoxToken(
                    page=idx,
                    x0=float(w["x0"]),
                    y0=float(w["top"]),
                    x1=float(w["x1"]),
                    y1=float(w["bottom"]),
                    text=str(w["text"]),
                )
                for w in raw_words
                if str(w.get("text", "")).strip()
            ]
            tokens_by_page[idx] = page_tokens
    return tokens_by_page, total_pages


def _row_text(row: Sequence[BBoxToken]) -> str:
    return normalize_text(" ".join(t.text for t in row))


def _is_footer_noise(row: Sequence[BBoxToken]) -> bool:
    text = _row_text(row)
    return bool(_FOOTER_NOISE_RE.search(text))


def _extract_header_metadata(
    page1_rows: list[list[BBoxToken]],
    *,
    source: str,
) -> tuple[GroundedField[str], GroundedField[str], GroundedField[str] | None]:
    invoice_no_field: GroundedField[str] | None = None
    invoice_date_field: GroundedField[str] | None = None
    vendor_field: GroundedField[str] | None = None

    # Vendor name: topmost left-column tokens on page 1
    for row in page1_rows[:4]:
        left_toks = [t for t in row if t.x0 < 330 and t.text.upper() != "INVOICE"]
        if left_toks:
            vendor_text = normalize_text(" ".join(t.text for t in left_toks))
            vendor_field = GroundedField(
                vendor_text,
                build_provenance(left_toks, source=source),
            )
            break

    for row in page1_rows:
        lower_tokens = [t.text.lower().rstrip(":") for t in row]
        for idx, word in enumerate(lower_tokens):
            if (
                word == "invoice"
                and idx + 1 < len(lower_tokens)
                and lower_tokens[idx + 1] in {"no", "#", "number"}
            ):
                right_toks = [t for t in row[idx + 2 :] if t.x0 > row[idx + 1].x1]
                if right_toks and invoice_no_field is None:
                    inv_tok = right_toks[0]
                    invoice_no_field = GroundedField(
                        inv_tok.text.strip(),
                        build_provenance([inv_tok], source=source),
                    )
            if (
                word == "invoice"
                and idx + 1 < len(lower_tokens)
                and lower_tokens[idx + 1] == "date"
            ):
                right_toks = [t for t in row[idx + 2 :] if t.x0 > row[idx + 1].x1]
                if right_toks and invoice_date_field is None:
                    date_tok = right_toks[0]
                    invoice_date_field = GroundedField(
                        date_tok.text.strip(),
                        build_provenance([date_tok], source=source),
                    )

    # Regex fallback within top half of Page 1 if labels were split or missing
    if invoice_no_field is None or invoice_date_field is None:
        for row in page1_rows:
            if row and row[0].y0 > 350:
                continue
            for tok in row:
                if invoice_no_field is None and _INV_NO_RE.match(tok.text.strip()):
                    invoice_no_field = GroundedField(
                        tok.text.strip(),
                        build_provenance([tok], source=source),
                        confidence=0.85,
                    )
                if invoice_date_field is None and _DATE_RE.match(tok.text.strip()):
                    invoice_date_field = GroundedField(
                        tok.text.strip(),
                        build_provenance([tok], source=source),
                        confidence=0.85,
                    )

    if invoice_no_field is None:
        invoice_no_field = GroundedField("", None, confidence=0.0)
    if invoice_date_field is None:
        invoice_date_field = GroundedField("", None, confidence=0.0)
    return invoice_no_field, invoice_date_field, vendor_field


@dataclass(frozen=True)
class _SummaryExtract:
    subtotal: GroundedField[int]
    credits: GroundedField[int]
    tax: GroundedField[int]
    total: GroundedField[int]
    summary_start_y: float


def _extract_summary_block(
    last_page_rows: list[list[BBoxToken]],
    *,
    source: str,
) -> _SummaryExtract:
    subtotal_f: GroundedField[int] | None = None
    credits_f: GroundedField[int] | None = None
    tax_f: GroundedField[int] | None = None
    total_f: GroundedField[int] | None = None
    summary_y_candidates: list[float] = []

    # Scan bottom-up so the invoice-level summary block takes precedence over group subtotals
    for row in reversed(last_page_rows):
        if _is_footer_noise(row):
            continue
        money_toks = [t for t in row if parse_currency_cents(t.text) is not None]
        if not money_toks:
            continue
        val_tok = money_toks[-1]
        val_cents = parse_currency_cents(val_tok.text)
        if val_cents is None:
            continue

        label_toks = [t for t in row if t.x1 < val_tok.x0 - 2.0]
        label_str = normalize_text(" ".join(t.text for t in label_toks)).lower().rstrip(":")

        if total_f is None and any(
            k in label_str for k in ("grand total", "total due", "amount due")
        ):
            total_f = GroundedField(val_cents, build_provenance([val_tok], source=source))
            summary_y_candidates.append(min(t.y0 for t in row))
        elif tax_f is None and ("tax" in label_str and "tax id" not in label_str):
            tax_f = GroundedField(val_cents, build_provenance([val_tok], source=source))
            summary_y_candidates.append(min(t.y0 for t in row))
        elif credits_f is None and any(
            k in label_str for k in ("credits applied", "less credits", "credit applied")
        ):
            credits_f = GroundedField(abs(val_cents), build_provenance([val_tok], source=source))
            summary_y_candidates.append(min(t.y0 for t in row))
        elif subtotal_f is None and label_str in {
            "subtotal",
            "sub-total",
            "sub total",
            "net subtotal",
            "invoice subtotal",
        }:
            subtotal_f = GroundedField(val_cents, build_provenance([val_tok], source=source))
            summary_y_candidates.append(min(t.y0 for t in row))
            # Once summary subtotal is reached scanning bottom-up, stop summary scan
            if total_f is not None:
                break

    start_y = min(summary_y_candidates) - 8.0 if summary_y_candidates else 10_000.0
    return _SummaryExtract(
        subtotal=subtotal_f or GroundedField(0, None, confidence=0.0),
        credits=credits_f or GroundedField(0, None, confidence=1.0),
        tax=tax_f or GroundedField(0, None, confidence=0.0),
        total=total_f or GroundedField(0, None, confidence=0.0),
        summary_start_y=start_y,
    )


def _cluster_header_columns(tokens: Sequence[BBoxToken]) -> list[list[BBoxToken]]:
    """Group header band tokens into horizontal columns even across 2-line headers."""
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: t.x0)
    clusters: list[list[BBoxToken]] = [[ordered[0]]]
    for tok in ordered[1:]:
        matched_idx: int | None = None
        for idx, grp in enumerate(clusters):
            grp_x0 = min(t.x0 for t in grp)
            grp_x1 = max(t.x1 for t in grp)
            # Same column if horizontally overlapping or close
            if tok.x0 <= grp_x1 + 14.0 and tok.x1 >= grp_x0 - 14.0:
                matched_idx = idx
                break
        if matched_idx is not None:
            clusters[matched_idx].append(tok)
        else:
            clusters.append([tok])
    clusters.sort(key=lambda grp: min(t.x0 for t in grp))
    return clusters


def _classify_header_cluster(cluster: Sequence[BBoxToken]) -> str | None:
    words = [t.text.lower().strip(":.") for t in cluster]
    word_set = set(words)
    joined = " ".join(words)

    if any(w in _AMOUNT_HDR_WORDS for w in word_set) or "ext'd" in joined:
        return "amount"
    if any(w in _UNIT_PRICE_HDR_WORDS for w in word_set):
        return "unit_price"
    if any(w in _QTY_HDR_WORDS for w in word_set):
        return "qty"
    if any(w in _DESC_HDR_WORDS for w in word_set):
        return "desc"
    if any(w in _SKU_HDR_WORDS for w in word_set):
        return "sku"
    return None


def _detect_header_in_rows(
    rows: list[list[BBoxToken]],
) -> tuple[int, int, dict[str, list[BBoxToken]]] | None:
    """Detect a single-line or 2-line wrapped column header band in page rows."""
    for idx, row in enumerate(rows):
        if _is_footer_noise(row):
            continue
        # Test 2-line header first if next row is vertically adjacent
        candidates: list[tuple[int, list[BBoxToken]]] = []
        if idx + 1 < len(rows) and not _is_footer_noise(rows[idx + 1]):
            y_gap = min(t.y0 for t in rows[idx + 1]) - max(t.y1 for t in row)
            if -2.0 <= y_gap <= 10.0:
                candidates.append((idx + 1, [*row, *rows[idx + 1]]))
        candidates.append((idx, list(row)))

        for end_idx, band_tokens in candidates:
            clusters = _cluster_header_columns(band_tokens)
            role_map: dict[str, list[BBoxToken]] = {}
            for cluster in clusters:
                role = _classify_header_cluster(cluster)
                if role is not None and role not in role_map:
                    role_map[role] = cluster
            if (
                "amount" in role_map
                and ("desc" in role_map or "sku" in role_map)
                and ("qty" in role_map or "unit_price" in role_map)
                and len(role_map) >= 4
            ):
                # Prefer 1-line match unless row idx+1 actually contributed header tokens
                if end_idx > idx:
                    row_next_words = {t.text.lower().strip(":.") for t in rows[idx + 1]}
                    header_vocab = (
                        _SKU_HDR_WORDS
                        | _DESC_HDR_WORDS
                        | _QTY_HDR_WORDS
                        | _UNIT_PRICE_HDR_WORDS
                        | _AMOUNT_HDR_WORDS
                    )
                    if not (row_next_words & header_vocab):
                        continue
                return idx, end_idx, role_map
    return None


def _refine_right_anchor(
    initial_x1: float,
    numeric_tokens: Sequence[BBoxToken],
    *,
    radius: float = 28.0,
) -> float:
    nearby = [t.x1 for t in numeric_tokens if abs(t.x1 - initial_x1) <= radius]
    if not nearby:
        return initial_x1
    return float(median(nearby))


def infer_table_columns(
    rows: list[list[BBoxToken]],
    *,
    y_limit: float,
) -> tuple[int, TableColumns] | None:
    """Detect header band and refine column boundaries using body numeric x1 clustering."""
    detected = _detect_header_in_rows(rows)
    if detected is None:
        return None
    _, header_end_idx, role_map = detected
    header_end_y = max(t.y1 for t in rows[header_end_idx])

    sku_toks = role_map.get("sku", [])
    desc_toks = role_map.get("desc", [])
    qty_toks = role_map.get("qty", [])
    unit_toks = role_map.get("unit_price", [])
    amt_toks = role_map["amount"]

    sku_x0 = min(t.x0 for t in sku_toks) if sku_toks else 46.0
    desc_x0 = min(t.x0 for t in desc_toks) if desc_toks else sku_x0 + 68.0
    qty_x1_init = max(t.x1 for t in qty_toks) if qty_toks else desc_x0 + 250.0
    unit_x1_init = max(t.x1 for t in unit_toks) if unit_toks else qty_x1_init + 80.0
    amt_x1_init = max(t.x1 for t in amt_toks)

    # Collect body numeric tokens below header_end_y to refine right-edge alignment
    body_numeric: list[BBoxToken] = []
    for row in rows[header_end_idx + 1 :]:
        if not row or min(t.y0 for t in row) >= y_limit or _is_footer_noise(row):
            continue
        for tok in row:
            if tok.x0 > desc_x0 + 30.0 and (
                parse_currency_cents(tok.text) is not None
                or parse_decimal_qty(tok.text) is not None
            ):
                body_numeric.append(tok)

    qty_x1 = _refine_right_anchor(qty_x1_init, body_numeric)
    unit_x1 = _refine_right_anchor(unit_x1_init, body_numeric)
    amt_x1 = _refine_right_anchor(amt_x1_init, body_numeric)

    # Left-aligned SKU values may extend toward desc_x0, so split just left of desc_x0
    sku_desc_split = desc_x0 - 6.0 if desc_x0 > sku_x0 + 20.0 else (sku_x0 + desc_x0) / 2.0
    desc_end_x = qty_x1 - 36.0

    return header_end_idx + 1, TableColumns(
        header_end_y=header_end_y,
        sku_x0=sku_x0,
        desc_x0=desc_x0,
        sku_desc_split=sku_desc_split,
        desc_end_x=desc_end_x,
        qty_anchor_x1=qty_x1,
        unit_price_anchor_x1=unit_x1,
        amount_anchor_x1=amt_x1,
    )


def _assign_numeric_column(tok: BBoxToken, cols: TableColumns) -> str | None:
    dist_qty = abs(tok.x1 - cols.qty_anchor_x1)
    dist_unit = abs(tok.x1 - cols.unit_price_anchor_x1)
    dist_amt = abs(tok.x1 - cols.amount_anchor_x1)
    best_col, best_dist = min(
        (("qty", dist_qty), ("unit_price", dist_unit), ("amount", dist_amt)),
        key=lambda pair: pair[1],
    )
    if best_dist > 36.0:
        return None
    if best_col == "qty" and parse_decimal_qty(tok.text) is None:
        return None
    if best_col in {"unit_price", "amount"} and parse_currency_cents(tok.text) is None:
        return None
    return best_col


def _parse_table_rows(
    page: int,
    rows: list[list[BBoxToken]],
    cols: TableColumns,
    *,
    y_limit: float,
    active_item: _CandidateRow | None,
) -> tuple[list[_CandidateRow], _CandidateRow | None]:
    candidates: list[_CandidateRow] = []

    for row in rows:
        if not row or _is_footer_noise(row):
            continue
        if max(t.y1 for t in row) >= y_limit:
            continue

        sku_toks: list[BBoxToken] = []
        desc_toks: list[BBoxToken] = []
        qty_toks: list[BBoxToken] = []
        unit_toks: list[BBoxToken] = []
        amt_toks: list[BBoxToken] = []

        for tok in row:
            if tok.x0 >= cols.desc_end_x:
                col_name = _assign_numeric_column(tok, cols)
                if col_name == "qty":
                    qty_toks.append(tok)
                    continue
                if col_name == "unit_price":
                    unit_toks.append(tok)
                    continue
                if col_name == "amount":
                    amt_toks.append(tok)
                    continue
            if tok.x0 < cols.sku_desc_split:
                sku_toks.append(tok)
            else:
                desc_toks.append(tok)

        row_full_text = _row_text(row)

        # Row without any amount token: either section header or wrapped description continuation
        if not amt_toks and not unit_toks and not qty_toks:
            if _SECTION_HEADER_RE.match(row_full_text):
                active_item = None
                continue
            if not sku_toks and desc_toks and active_item is not None:
                active_item.desc_tokens.extend(desc_toks)
            elif sku_toks and not desc_toks:
                # Section title that spans into sku region only
                active_item = None
            continue

        if not amt_toks:
            continue

        amt_cents = parse_currency_cents(amt_toks[-1].text)
        if amt_cents is None:
            continue

        if _SUBTOTAL_LABEL_RE.search(row_full_text):
            active_item = None
            candidates.append(
                _CandidateRow(
                    page=page,
                    kind="subtotal_label",
                    sku_tokens=sku_toks,
                    desc_tokens=desc_toks,
                    amount_tokens=amt_toks,
                    amount_cents=amt_cents,
                )
            )
            continue

        qty_val = parse_decimal_qty(qty_toks[-1].text) if qty_toks else None
        unit_cents = parse_currency_cents(unit_toks[-1].text) if unit_toks else None

        if qty_val is not None and unit_cents is not None and desc_toks:
            expected_cents = int((qty_val * unit_cents).to_integral_value())
            kind = "definite" if expected_cents == amt_cents else "ambiguous"
            item_row = _CandidateRow(
                page=page,
                kind=kind,
                sku_tokens=sku_toks,
                desc_tokens=desc_toks,
                qty_tokens=qty_toks,
                unit_price_tokens=unit_toks,
                amount_tokens=amt_toks,
                qty_val=qty_val,
                unit_price_cents=unit_cents,
                amount_cents=amt_cents,
            )
            candidates.append(item_row)
            active_item = item_row
        else:
            # Ambiguous row with amount but missing qty/unit_price (flat fee or unlabeled subtotal)
            active_item = None
            candidates.append(
                _CandidateRow(
                    page=page,
                    kind="ambiguous",
                    sku_tokens=sku_toks,
                    desc_tokens=desc_toks,
                    qty_tokens=qty_toks,
                    unit_price_tokens=unit_toks,
                    amount_tokens=amt_toks,
                    qty_val=qty_val or Decimal("1"),
                    unit_price_cents=unit_cents if unit_cents is not None else amt_cents,
                    amount_cents=amt_cents,
                )
            )

    return candidates, active_item


def find_exact_subset_indices(
    amounts: Sequence[int],
    target: int,
    *,
    max_states: int = 50_000,
) -> list[int] | None:
    """Find indices of `amounts` summing exactly to `target`, preferring larger cardinality."""
    if target == 0:
        return []
    n = len(amounts)
    if n == 0:
        return None

    # Suffix positive/negative bounds for branch-and-bound pruning
    pos_suffix = [0] * (n + 1)
    neg_suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        val = amounts[i]
        pos_suffix[i] = pos_suffix[i + 1] + (val if val > 0 else 0)
        neg_suffix[i] = neg_suffix[i + 1] + (val if val < 0 else 0)

    best: list[int] | None = None
    steps = 0

    def dfs(idx: int, current_sum: int, chosen: list[int]) -> None:
        nonlocal best, steps
        steps += 1
        if steps > max_states:
            return
        if current_sum + pos_suffix[idx] < target or current_sum + neg_suffix[idx] > target:
            return
        if idx == n:
            if current_sum == target and (best is None or len(chosen) > len(best)):
                best = list(chosen)
            return

        # Try including idx first to favor keeping genuine line items
        chosen.append(idx)
        dfs(idx + 1, current_sum + amounts[idx], chosen)
        chosen.pop()
        if steps <= max_states:
            dfs(idx + 1, current_sum, chosen)

    dfs(0, 0, [])
    return best


def disambiguate_candidate_rows(
    candidates: Sequence[_CandidateRow],
    target_subtotal_cents: int,
) -> tuple[list[_CandidateRow], list[str]]:
    """Resolve definite items vs ambiguous/subtotal rows using exact-cent subset-sum."""
    warnings: list[str] = []
    definite = [r for r in candidates if r.kind == "definite"]
    ambiguous = [r for r in candidates if r.kind == "ambiguous"]
    subtotals = [r for r in candidates if r.kind == "subtotal_label"]

    definite_sum = sum(r.amount_cents for r in definite)
    if definite_sum == target_subtotal_cents:
        if subtotals:
            warnings.append(f"filtered {len(subtotals)} group subtotal row(s)")
        return definite, warnings

    # Case 2: definite rows fall short and ambiguous rows may hold flat-fee items vs subtotals
    if ambiguous:
        residual = target_subtotal_cents - definite_sum
        subset = find_exact_subset_indices([r.amount_cents for r in ambiguous], residual)
        if subset is not None:
            chosen_ids = {id(ambiguous[idx]) for idx in subset}
            resolved = [
                r
                for r in candidates
                if r.kind == "definite" or (r.kind == "ambiguous" and id(r) in chosen_ids)
            ]
            dropped = len(ambiguous) - len(subset) + len(subtotals)
            warnings.append(
                f"subset-sum disambiguation kept {len(subset)} ambiguous item(s), "
                f"excluded {dropped} subtotal/summary row(s)"
            )
            return resolved, warnings

    # Case 3: definite rows overshot target (e.g. an unlabeled subtotal looked like a line item)
    pool = [r for r in candidates if r.kind != "subtotal_label"]
    subset_all = find_exact_subset_indices([r.amount_cents for r in pool], target_subtotal_cents)
    if subset_all is not None:
        chosen_all = set(subset_all)
        resolved_all = [r for i, r in enumerate(pool) if i in chosen_all]
        warnings.append(f"subset-sum pruned {len(pool) - len(resolved_all)} over-counted row(s)")
        return resolved_all, warnings

    return [*definite, *ambiguous], warnings


def parse_invoice(
    pdf_path: Path,
    *,
    fallback_provider: FallbackProvider | None = None,
    layout_hint: str | None = None,
) -> InvoiceResult:
    """Extract structured line items & header fields from a PDF invoice with exact grounding."""
    t0 = time.perf_counter()
    source = "deterministic_geometry"

    tokens_by_page, total_pages = extract_tokens(pdf_path)
    rows_by_page = {page: cluster_tokens_into_rows(toks) for page, toks in tokens_by_page.items()}

    page1_rows = rows_by_page.get(1, [])
    inv_no_f, inv_date_f, vendor_f = _extract_header_metadata(page1_rows, source=source)

    last_rows = rows_by_page.get(total_pages, [])
    summary = _extract_summary_block(last_rows, source=source)

    all_candidates: list[_CandidateRow] = []
    active_item: _CandidateRow | None = None
    prev_cols: TableColumns | None = None
    warnings: list[str] = []

    for page in range(1, total_pages + 1):
        p_rows = rows_by_page.get(page, [])
        y_limit = summary.summary_start_y if page == total_pages else 10_000.0
        detected = infer_table_columns(p_rows, y_limit=y_limit)
        if detected is not None:
            start_idx, cols = detected
            prev_cols = cols
        elif prev_cols is not None:
            start_idx = 0
            cols = prev_cols
        else:
            warnings.append(f"page {page}: table header not detected")
            continue

        page_cands, active_item = _parse_table_rows(
            page,
            p_rows[start_idx:],
            cols,
            y_limit=y_limit,
            active_item=active_item,
        )
        all_candidates.extend(page_cands)

    resolved_rows, disamb_warnings = disambiguate_candidate_rows(
        all_candidates, summary.subtotal.value
    )
    warnings.extend(disamb_warnings)

    line_items = [row.to_line_item(source=source) for row in resolved_rows]
    line_sum = sum(item.amount_cents.value for item in line_items)
    expected_grand = summary.subtotal.value + summary.tax.value - summary.credits.value

    checksum_passed = (
        bool(line_items)
        and bool(inv_no_f.value)
        and bool(inv_date_f.value)
        and summary.subtotal.provenance is not None
        and summary.tax.provenance is not None
        and summary.total.provenance is not None
        and line_sum == summary.subtotal.value
        and expected_grand == summary.total.value
    )

    checksum_detail = (
        f"line_sum={line_sum} {'==' if line_sum == summary.subtotal.value else '!='} "
        f"subtotal={summary.subtotal.value}; "
        f"subtotal+tax-credits={expected_grand} "
        f"{'==' if expected_grand == summary.total.value else '!='} "
        f"total={summary.total.value}"
    )

    result = InvoiceResult(
        invoice_no=inv_no_f,
        invoice_date=inv_date_f,
        subtotal_cents=summary.subtotal,
        tax_cents=summary.tax,
        credits_cents=summary.credits,
        total_cents=summary.total,
        line_items=line_items,
        checksum_passed=checksum_passed,
        checksum_detail=checksum_detail,
        vendor_name=vendor_f,
        currency="USD",
        layout_hint=layout_hint,
        pages=total_pages,
        warnings=warnings,
    )

    # Strict gate: only invoke fallback when deterministic checksum fails
    if not result.checksum_passed and fallback_provider is not None:
        result = apply_strict_fallback(result, tokens_by_page, fallback_provider)

    result.latency_ms = (time.perf_counter() - t0) * 1000.0
    return result


def ground_check(result: InvoiceResult, pdf_path: Path) -> Provenance | None:
    """Helper returning first missing/invalid provenance if any, else None."""
    tokens_by_page, _ = extract_tokens(pdf_path)
    for prov in result.header_provenances().values():
        if prov is None or not prov.is_valid():
            return prov
    for item in result.line_items:
        for prov in item.field_provenances().values():
            if prov is None or not prov.is_valid() or prov.page not in tokens_by_page:
                return prov
    return None
