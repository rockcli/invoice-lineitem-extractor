"""Token bounding-box provenance construction and verification utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from invoice_extractor.models import BBoxToken, InvoiceResult, Provenance, union_bbox

TokenId = tuple[int, int]

_CURRENCY_NUMBER_RE = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?"
_CURRENCY_CLEAN_RE = re.compile(
    rf"^(?:\([-+]?\$?{_CURRENCY_NUMBER_RE}\)|[-+]?\$?{_CURRENCY_NUMBER_RE})$"
)
_SIMPLE_DECIMAL_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def parse_currency_cents(text: str) -> int | None:
    """Parse a currency token like '$1,240.00', '($45.00)', or '-45.00' to exact integer cents."""
    cleaned = text.strip()
    if not cleaned:
        return None
    # Strip trailing non-numeric punctuation occasionally attached in OCR footers
    if cleaned.endswith((",", ";")):
        cleaned = cleaned[:-1].strip()
    if not _CURRENCY_CLEAN_RE.match(cleaned):
        return None

    is_negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        is_negative = True
        cleaned = cleaned[1:-1].strip()

    cleaned = cleaned.replace("$", "").replace(",", "").strip()
    if cleaned.startswith("-"):
        is_negative = True
        cleaned = cleaned[1:].strip()
    elif cleaned.startswith("+"):
        cleaned = cleaned[1:].strip()

    if not cleaned or not re.match(r"^\d+(?:\.\d{2})?$", cleaned):
        return None

    try:
        dec = Decimal(cleaned)
    except InvalidOperation:
        return None

    cents = int((dec * 100).to_integral_value())
    return -cents if is_negative else cents


def parse_decimal_qty(text: str) -> Decimal | None:
    """Parse a numeric quantity string like '2' or '1.5' without currency symbols."""
    cleaned = text.strip().replace(",", "")
    if not cleaned or "$" in cleaned or "(" in cleaned or ")" in cleaned:
        return None
    if not _SIMPLE_DECIMAL_RE.match(cleaned):
        return None
    try:
        val = Decimal(cleaned)
    except InvalidOperation:
        return None
    if val <= 0 or val > Decimal("100000"):
        return None
    return val


def normalize_text(text: str) -> str:
    """Collapse internal whitespace for stable provenance text matching."""
    return " ".join(text.split())


def cluster_tokens_into_rows(
    tokens: Sequence[BBoxToken],
    *,
    y_tol: float = 5.2,
) -> list[list[BBoxToken]]:
    """Group tokens into visual horizontal rows tolerant of ±2.5pt vertical OCR jitter."""
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda t: (t.cy, t.x0))
    rows: list[list[BBoxToken]] = []
    current_row: list[BBoxToken] = [ordered[0]]
    current_mean_y = ordered[0].cy

    for tok in ordered[1:]:
        if abs(tok.cy - current_mean_y) <= y_tol:
            current_row.append(tok)
            current_mean_y = sum(t.cy for t in current_row) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda t: t.x0))
            current_row = [tok]
            current_mean_y = tok.cy
    if current_row:
        rows.append(sorted(current_row, key=lambda t: t.x0))
    return rows


def flatten_reading_order(tokens: Sequence[BBoxToken], *, y_tol: float = 5.2) -> list[BBoxToken]:
    """Return tokens ordered top-to-bottom by visual row, then left-to-right."""
    return [tok for row in cluster_tokens_into_rows(tokens, y_tol=y_tol) for tok in row]


def build_provenance(
    tokens: Sequence[BBoxToken],
    *,
    source: str,
    raw_text_override: str | None = None,
) -> Provenance:
    """Construct a Provenance object spanning one or more same-page tokens."""
    if not tokens:
        raise ValueError("Cannot build Provenance from empty token sequence.")
    page = tokens[0].page
    if any(t.page != page for t in tokens):
        raise ValueError("All tokens in a single Provenance span must share the same page.")
    ordered = flatten_reading_order(tokens)
    bbox = union_bbox([t.bbox for t in ordered])
    raw_text = (
        raw_text_override if raw_text_override is not None else " ".join(t.text for t in ordered)
    )
    return Provenance(page=page, bbox=bbox, raw_text=raw_text, source=source)


def tokens_in_bbox(
    page_tokens: Sequence[BBoxToken],
    bbox: tuple[float, float, float, float],
    *,
    pad: float = 0.8,
) -> list[BBoxToken]:
    """Return all tokens on a page intersecting `bbox` within `pad` points."""
    x0, y0, x1, y1 = bbox
    return [
        tok
        for tok in page_tokens
        if tok.x1 >= x0 - pad and tok.x0 <= x1 + pad and tok.y1 >= y0 - pad and tok.y0 <= y1 + pad
    ]


def verify_provenance(
    provenance: Provenance | None,
    tokens_by_page: dict[int, list[BBoxToken]],
    *,
    pad: float = 1.0,
) -> bool:
    """Verify that `provenance` points to valid coordinates and matching PDF tokens."""
    if provenance is None or not provenance.is_valid():
        return False
    page_tokens = tokens_by_page.get(provenance.page)
    if not page_tokens:
        return False
    overlapping = tokens_in_bbox(page_tokens, provenance.bbox, pad=pad)
    if not overlapping:
        return False

    expected = normalize_text(provenance.raw_text)
    observed = normalize_text(" ".join(t.text for t in overlapping))
    if expected == observed or expected in observed:
        return True

    # Fallback check: every whitespace-delimited token in raw_text appears in overlapping box
    obs_words = set(observed.split())
    exp_words = expected.split()
    return bool(exp_words) and all(word in obs_words for word in exp_words)


def verify_line_item_arithmetic(
    qty: Decimal,
    unit_price_cents: int,
    amount_cents: int,
) -> bool:
    """Verify exact cent line arithmetic (exact for integer qty, <=1 cent for fractional qty)."""
    expected_cents = int(
        (qty * Decimal(unit_price_cents)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    tolerance = 0 if qty == qty.to_integral_value() else 1
    return abs(expected_cents - amount_cents) <= tolerance


def tokens_row_colocated(tokens: Sequence[BBoxToken], *, row_tol: float = 8.0) -> bool:
    """Return True when all tokens lie on the same page within `row_tol` vertical midpoint delta."""
    if not tokens:
        return False
    page = tokens[0].page
    if any(t.page != page for t in tokens):
        return False
    y_mids = [t.cy for t in tokens]
    return (max(y_mids) - min(y_mids)) <= row_tol


def index_tokens_by_page(tokens_by_page: dict[int, list[BBoxToken]]) -> dict[int, list[BBoxToken]]:
    """Return reading-order normalized tokens per page for stable TokenId indexing."""
    return {page: flatten_reading_order(tokens_by_page[page]) for page in sorted(tokens_by_page)}


def find_cents_token_matches(
    target_cents: int,
    ordered_by_page: dict[int, list[BBoxToken]],
    *,
    excluded: set[TokenId] | None = None,
    reverse_scan: bool = False,
) -> list[tuple[TokenId, BBoxToken]]:
    """Return all `(TokenId, BBoxToken)` pairs matching `target_cents` not in `excluded`."""
    blocked = excluded or set()
    matches: list[tuple[TokenId, BBoxToken]] = []
    pages = sorted(ordered_by_page, reverse=reverse_scan)
    for page in pages:
        toks = ordered_by_page[page]
        indices = range(len(toks) - 1, -1, -1) if reverse_scan else range(len(toks))
        for idx in indices:
            tok_id = (page, idx)
            if tok_id in blocked:
                continue
            tok = toks[idx]
            parsed = parse_currency_cents(tok.text)
            if parsed is not None and parsed == target_cents:
                matches.append((tok_id, tok))
    return matches


def find_qty_token_matches_on_page(
    target_qty: Decimal,
    page: int,
    page_tokens: Sequence[BBoxToken],
    *,
    excluded: set[TokenId] | None = None,
    anchor_cy: float | None = None,
    row_tol: float = 8.0,
) -> list[tuple[TokenId, BBoxToken]]:
    """Return `(TokenId, BBoxToken)` pairs on `page` matching `target_qty`."""
    blocked = excluded or set()
    matches: list[tuple[TokenId, BBoxToken]] = []
    for idx, tok in enumerate(page_tokens):
        tok_id = (page, idx)
        if tok_id in blocked:
            continue
        if anchor_cy is not None and abs(tok.cy - anchor_cy) > row_tol:
            continue
        parsed = parse_decimal_qty(tok.text)
        if parsed is not None and parsed == target_qty:
            matches.append((tok_id, tok))
    return matches


def find_cents_token_matches_on_page(
    target_cents: int,
    page: int,
    page_tokens: Sequence[BBoxToken],
    *,
    excluded: set[TokenId] | None = None,
    anchor_cy: float | None = None,
    row_tol: float = 8.0,
) -> list[tuple[TokenId, BBoxToken]]:
    """Return all `(TokenId, BBoxToken)` on `page` matching `target_cents`."""
    blocked = excluded or set()
    matches: list[tuple[TokenId, BBoxToken]] = []
    for idx, tok in enumerate(page_tokens):
        tok_id = (page, idx)
        if tok_id in blocked:
            continue
        if anchor_cy is not None and abs(tok.cy - anchor_cy) > row_tol:
            continue
        parsed = parse_currency_cents(tok.text)
        if parsed is not None and parsed == target_cents:
            matches.append((tok_id, tok))
    return matches


def find_text_span_matches_on_page(
    target_text: str,
    page: int,
    page_tokens: Sequence[BBoxToken],
    *,
    excluded: set[TokenId] | None = None,
    anchor_cy: float | None = None,
    row_tol: float = 8.0,
) -> list[tuple[list[TokenId], list[BBoxToken]]]:
    """Return contiguous token spans on `page` matching `target_text`."""
    target_norm = normalize_text(target_text)
    if not target_norm:
        return []
    target_parts = target_norm.split()
    n_parts = len(target_parts)
    blocked = excluded or set()
    matches: list[tuple[list[TokenId], list[BBoxToken]]] = []

    for start in range(len(page_tokens) - n_parts + 1):
        ids = [(page, start + offset) for offset in range(n_parts)]
        if any(tok_id in blocked for tok_id in ids):
            continue
        window = list(page_tokens[start : start + n_parts])
        if anchor_cy is not None and any(abs(t.cy - anchor_cy) > row_tol for t in window):
            continue
        if [normalize_text(t.text) for t in window] == target_parts:
            matches.append((ids, window))
    return matches


def find_wrapped_text_matches_on_page(
    target_text: str,
    page: int,
    page_tokens: Sequence[BBoxToken],
    *,
    excluded: set[TokenId] | None = None,
    skippable_ids: set[TokenId] | None = None,
    anchor_cy: float,
    row_tol: float = 8.0,
    max_wrap_dy: float = 18.0,
) -> list[tuple[list[TokenId], list[BBoxToken]]]:
    """Locate single-line or line-wrapped description tokens anchored at `anchor_cy`."""
    words = normalize_text(target_text).split()
    if not words:
        return []
    blocked = excluded or set()
    skippable = skippable_ids or set()
    matches: list[tuple[list[TokenId], list[BBoxToken]]] = []

    for start in range(len(page_tokens)):
        start_id = (page, start)
        if start_id in blocked or start_id in skippable:
            continue
        start_tok = page_tokens[start]
        if abs(start_tok.cy - anchor_cy) > row_tol:
            continue
        if normalize_text(start_tok.text) != words[0]:
            continue

        matched_ids: list[TokenId] = [start_id]
        matched_toks: list[BBoxToken] = [start_tok]
        cursor = 1
        prev_tok = start_tok

        for idx in range(start + 1, len(page_tokens)):
            if cursor >= len(words):
                break
            tok_id = (page, idx)
            if tok_id in skippable:
                continue
            if tok_id in blocked:
                break
            tok = page_tokens[idx]
            dy = tok.cy - prev_tok.cy
            if dy < -row_tol or dy > max_wrap_dy:
                break
            if normalize_text(tok.text) != words[cursor]:
                break
            matched_ids.append(tok_id)
            matched_toks.append(tok)
            prev_tok = tok
            cursor += 1

        if cursor == len(words):
            matches.append((matched_ids, matched_toks))
    return matches


def find_text_token_span(
    target_text: str,
    tokens_by_page: dict[int, list[BBoxToken]],
    *,
    source: str,
) -> Provenance | None:
    """Locate an exact or contiguous multi-token text match in PDF tokens."""
    target_norm = normalize_text(target_text)
    if not target_norm:
        return None
    target_parts = target_norm.split()
    n_parts = len(target_parts)

    for page in sorted(tokens_by_page):
        toks = flatten_reading_order(tokens_by_page[page])
        if n_parts == 1:
            for tok in toks:
                if normalize_text(tok.text) == target_norm:
                    return build_provenance([tok], source=source)
        for idx in range(len(toks) - n_parts + 1):
            window = toks[idx : idx + n_parts]
            if normalize_text(" ".join(t.text for t in window)) == target_norm:
                return build_provenance(window, source=source)
    return None


def find_cents_token(
    target_cents: int,
    tokens_by_page: dict[int, list[BBoxToken]],
    *,
    source: str,
) -> Provenance | None:
    """Locate a PDF token whose parsed currency cents exactly equals `target_cents`."""
    for page in sorted(tokens_by_page):
        for tok in tokens_by_page[page]:
            parsed = parse_currency_cents(tok.text)
            if parsed is not None and parsed == target_cents:
                return build_provenance([tok], source=source)
    return None


def find_qty_token(
    target_qty: Decimal,
    tokens_by_page: dict[int, list[BBoxToken]],
    *,
    source: str,
) -> Provenance | None:
    """Locate a PDF token whose parsed decimal quantity equals `target_qty`."""
    for page in sorted(tokens_by_page):
        for tok in tokens_by_page[page]:
            parsed = parse_decimal_qty(tok.text)
            if parsed is not None and parsed == target_qty:
                return build_provenance([tok], source=source)
    return None


@dataclass(frozen=True)
class GroundingAudit:
    """Summary of provenance verification across all fields of an InvoiceResult."""

    total_fields: int
    grounded_fields: int
    failed_fields: tuple[str, ...]

    @property
    def validity_rate(self) -> float:
        if self.total_fields == 0:
            return 0.0
        return self.grounded_fields / self.total_fields


def verify_invoice_grounding(
    result: InvoiceResult,
    tokens_by_page: dict[int, list[BBoxToken]],
) -> GroundingAudit:
    """Verify provenance on every header field and every line-item field."""
    failed: list[str] = []
    total = 0
    grounded = 0

    for name, prov in result.header_provenances().items():
        total += 1
        if verify_provenance(prov, tokens_by_page):
            grounded += 1
        else:
            failed.append(f"header.{name}")

    for idx, item in enumerate(result.line_items):
        for field_name, prov in item.field_provenances().items():
            total += 1
            if verify_provenance(prov, tokens_by_page):
                grounded += 1
            else:
                failed.append(f"line_items[{idx}].{field_name}")

    return GroundingAudit(
        total_fields=total,
        grounded_fields=grounded,
        failed_fields=tuple(failed),
    )
