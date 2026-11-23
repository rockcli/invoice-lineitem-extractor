"""Strict-gated, token-grounded LLM fallback interface for checksum failures."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from invoice_extractor.grounding import (
    TokenId,
    build_provenance,
    cluster_tokens_into_rows,
    find_cents_token_matches,
    find_cents_token_matches_on_page,
    find_qty_token_matches_on_page,
    find_text_span_matches_on_page,
    find_wrapped_text_matches_on_page,
    index_tokens_by_page,
    is_wrapped_description_span_clean,
    normalize_text,
    tokens_row_colocated,
    verify_line_item_arithmetic,
)
from invoice_extractor.models import (
    BBoxToken,
    GroundedField,
    InvoiceResult,
    LineItem,
    Provenance,
    union_bbox,
)


class FallbackProvider(Protocol):
    """Pluggable repair proposal interface invoked ONLY on deterministic checksum failure."""

    name: str

    def propose_repair(
        self,
        result: InvoiceResult,
        tokens_by_page: dict[int, list[BBoxToken]],
    ) -> dict[str, Any] | None:
        """Return a structured JSON proposal matching the strict invoice schema, or None."""
        ...


class DisabledFallbackProvider:
    """Default offline no-op provider ensuring zero external network calls."""

    name = "disabled"

    def propose_repair(
        self,
        result: InvoiceResult,
        tokens_by_page: dict[int, list[BBoxToken]],
    ) -> dict[str, Any] | None:
        return None


class MockFallbackProvider:
    """Deterministic offline mock provider for testing fallback gating and grounding."""

    name = "mock"

    def __init__(
        self,
        payload: dict[str, Any]
        | Callable[[InvoiceResult, dict[int, list[BBoxToken]]], dict[str, Any] | None]
        | None = None,
    ) -> None:
        self._payload = payload
        self.call_count = 0

    def propose_repair(
        self,
        result: InvoiceResult,
        tokens_by_page: dict[int, list[BBoxToken]],
    ) -> dict[str, Any] | None:
        self.call_count += 1
        if callable(self._payload):
            return self._payload(result, tokens_by_page)
        return self._payload


def resolve_fallback_provider(mode: str | None = None) -> FallbackProvider:
    """Resolve offline fallback provider from explicit mode or INVOICE_EXTRACTOR_FALLBACK_MODE."""
    raw_mode = (
        mode if mode is not None else os.environ.get("INVOICE_EXTRACTOR_FALLBACK_MODE", "disabled")
    )
    normalized = raw_mode.strip().lower() or "disabled"
    if normalized == "disabled":
        return DisabledFallbackProvider()
    if normalized == "mock":
        return MockFallbackProvider()
    raise ValueError(
        f"Unsupported INVOICE_EXTRACTOR_FALLBACK_MODE={raw_mode!r}; expected 'disabled' or 'mock'."
    )


_REQUIRED_TOP_KEYS = (
    "invoice_no",
    "invoice_date",
    "subtotal_cents",
    "tax_cents",
    "credits_cents",
    "total_cents",
    "line_items",
)
_REQUIRED_ITEM_KEYS = ("sku", "description", "qty", "unit_price_cents", "amount_cents")


def _validate_schema(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return "schema_validation_failed: payload must be a dict"
    for key in _REQUIRED_TOP_KEYS:
        if key not in payload:
            return f"schema_validation_failed: missing key '{key}'"

    for int_key in ("subtotal_cents", "tax_cents", "credits_cents", "total_cents"):
        val = payload[int_key]
        if isinstance(val, bool) or not isinstance(val, int):
            return f"schema_validation_failed: '{int_key}' must be integer cents"

    if not isinstance(payload["invoice_no"], str) or not payload["invoice_no"].strip():
        return "schema_validation_failed: invalid 'invoice_no'"
    if not isinstance(payload["invoice_date"], str) or not payload["invoice_date"].strip():
        return "schema_validation_failed: invalid 'invoice_date'"

    items = payload["line_items"]
    if not isinstance(items, list) or not items:
        return "schema_validation_failed: 'line_items' must be a non-empty list"

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            return f"schema_validation_failed: line_items[{idx}] must be a dict"
        for k in _REQUIRED_ITEM_KEYS:
            if k not in item:
                return f"schema_validation_failed: line_items[{idx}] missing '{k}'"
        for str_key in ("sku", "description"):
            if not isinstance(item[str_key], str) or not item[str_key].strip():
                return f"schema_validation_failed: line_items[{idx}].{str_key} invalid string"
        for int_key in ("unit_price_cents", "amount_cents"):
            v = item[int_key]
            if isinstance(v, bool) or not isinstance(v, int):
                return f"schema_validation_failed: line_items[{idx}].{int_key} must be int"
        try:
            qty = Decimal(str(item["qty"]))
        except (InvalidOperation, ValueError):
            return f"schema_validation_failed: line_items[{idx}].qty invalid decimal"
        if qty <= 0:
            return f"schema_validation_failed: line_items[{idx}].qty must be > 0"
    return None


def _ground_header_text(
    target_text: str,
    ordered_by_page: dict[int, list[BBoxToken]],
    *,
    used_tokens: set[TokenId],
    source: str,
) -> Provenance | None:
    for page in sorted(ordered_by_page):
        matches = find_text_span_matches_on_page(
            target_text,
            page,
            ordered_by_page[page],
            excluded=used_tokens,
        )
        if matches:
            ids, toks = matches[0]
            used_tokens.update(ids)
            return build_provenance(toks, source=source)
    return None


def _ground_summary_cents(
    target_cents: int,
    ordered_by_page: dict[int, list[BBoxToken]],
    *,
    used_tokens: set[TokenId],
    source: str,
) -> Provenance | None:
    matches = find_cents_token_matches(
        target_cents,
        ordered_by_page,
        excluded=used_tokens,
        reverse_scan=True,
    )
    if not matches:
        return None
    tok_id, tok = matches[0]
    used_tokens.add(tok_id)
    return build_provenance([tok], source=source)


def _diagnose_line_item_failure(
    *,
    idx: int,
    sku_str: str,
    desc_str: str,
    qty_val: Decimal,
    unit_cents: int,
    amt_cents: int,
    ordered_by_page: dict[int, list[BBoxToken]],
    used_tokens: set[TokenId],
) -> str:
    if not find_cents_token_matches(amt_cents, ordered_by_page, excluded=used_tokens):
        return f"ungrounded_line_items[{idx}].amount_cents: {amt_cents}"
    if not find_cents_token_matches(unit_cents, ordered_by_page):
        return f"ungrounded_line_items[{idx}].unit_price_cents: {unit_cents}"
    if not any(
        find_qty_token_matches_on_page(qty_val, page, toks)
        for page, toks in ordered_by_page.items()
    ):
        return f"ungrounded_line_items[{idx}].qty: {qty_val}"
    if not any(
        find_text_span_matches_on_page(sku_str, page, toks)
        for page, toks in ordered_by_page.items()
    ):
        return f"ungrounded_line_items[{idx}].sku: {sku_str}"
    if not any(
        find_wrapped_text_matches_on_page(
            desc_str,
            page,
            toks,
            anchor_cy=0.0,
            row_tol=100_000.0,
            max_wrap_dy=100_000.0,
            enforce_row_band=False,
        )
        for page, toks in ordered_by_page.items()
    ):
        return f"ungrounded_line_items[{idx}].description: {desc_str}"
    return (
        f"ungrounded_row_colocation_or_reused_token: line_items[{idx}] "
        "tokens are not co-located on a single unused row"
    )


def _ground_line_item_row(
    *,
    idx: int,
    sku_str: str,
    desc_str: str,
    qty_val: Decimal,
    unit_cents: int,
    amt_cents: int,
    ordered_by_page: dict[int, list[BBoxToken]],
    used_tokens: set[TokenId],
    source: str,
    row_tol: float,
) -> tuple[LineItem, set[TokenId]] | tuple[None, str]:
    amt_candidates = find_cents_token_matches(amt_cents, ordered_by_page, excluded=used_tokens)
    for amt_id, amt_tok in amt_candidates:
        page = amt_id[0]
        page_tokens = ordered_by_page[page]
        anchor_cy = amt_tok.cy
        base_excluded = used_tokens | {amt_id}

        unit_candidates = find_cents_token_matches_on_page(
            unit_cents,
            page,
            page_tokens,
            excluded=base_excluded,
            anchor_cy=anchor_cy,
            row_tol=row_tol,
        )
        qty_candidates = find_qty_token_matches_on_page(
            qty_val,
            page,
            page_tokens,
            excluded=base_excluded,
            anchor_cy=anchor_cy,
            row_tol=row_tol,
        )
        sku_candidates = find_text_span_matches_on_page(
            sku_str,
            page,
            page_tokens,
            excluded=base_excluded,
            anchor_cy=anchor_cy,
            row_tol=row_tol,
        )

        for unit_id, unit_tok in unit_candidates:
            for qty_id, qty_tok in qty_candidates:
                if qty_id == unit_id:
                    continue
                for sku_ids, sku_toks in sku_candidates:
                    sku_id_set = set(sku_ids)
                    if unit_id in sku_id_set or qty_id in sku_id_set:
                        continue
                    structural_toks = [amt_tok, unit_tok, qty_tok, *sku_toks]
                    if not tokens_row_colocated(structural_toks, row_tol=row_tol):
                        continue
                    row_structural_ids = {amt_id, unit_id, qty_id} | sku_id_set
                    desc_candidates = find_wrapped_text_matches_on_page(
                        desc_str,
                        page,
                        page_tokens,
                        excluded=used_tokens,
                        skippable_ids=row_structural_ids,
                        anchor_cy=anchor_cy,
                        row_tol=row_tol,
                    )
                    for desc_ids, desc_toks in desc_candidates:
                        anchor_line = cluster_tokens_into_rows(desc_toks, y_tol=5.2)[0]
                        if not tokens_row_colocated(
                            [*structural_toks, *anchor_line], row_tol=row_tol
                        ):
                            continue
                        if not is_wrapped_description_span_clean(
                            desc_ids,
                            desc_toks,
                            page_tokens,
                            page=page,
                            anchor_cy=anchor_cy,
                            row_tol=row_tol,
                            skippable_ids=row_structural_ids,
                        ):
                            continue
                        consumed = row_structural_ids | set(desc_ids)
                        all_toks = [*sku_toks, *desc_toks, qty_tok, unit_tok, amt_tok]
                        item = LineItem(
                            sku=GroundedField(
                                sku_str, build_provenance(sku_toks, source=source), confidence=0.9
                            ),
                            description=GroundedField(
                                desc_str, build_provenance(desc_toks, source=source), confidence=0.9
                            ),
                            qty=GroundedField(
                                qty_val, build_provenance([qty_tok], source=source), confidence=0.9
                            ),
                            unit_price_cents=GroundedField(
                                unit_cents,
                                build_provenance([unit_tok], source=source),
                                confidence=0.9,
                            ),
                            amount_cents=GroundedField(
                                amt_cents,
                                build_provenance([amt_tok], source=source),
                                confidence=0.9,
                            ),
                            page=page,
                            row_bbox=union_bbox([t.bbox for t in all_toks]),
                            is_credit=amt_cents < 0,
                        )
                        return item, consumed

    reason = _diagnose_line_item_failure(
        idx=idx,
        sku_str=sku_str,
        desc_str=desc_str,
        qty_val=qty_val,
        unit_cents=unit_cents,
        amt_cents=amt_cents,
        ordered_by_page=ordered_by_page,
        used_tokens=used_tokens,
    )
    return None, reason


def apply_strict_fallback(
    result: InvoiceResult,
    tokens_by_page: dict[int, list[BBoxToken]],
    provider: FallbackProvider | None,
    *,
    row_tol: float = 8.0,
) -> InvoiceResult:
    """Apply strict-gated fallback ONLY when `result.checksum_passed` is False.

    Enforces:
    1. Gate: Never invokes provider if deterministic checksum already passed.
    2. Strict JSON schema verification.
    3. Per-line arithmetic: `qty * unit_price_cents == amount_cents` (<=1c fractional rounding).
    4. Same-page horizontal row co-location across SKU/description/qty/unit_price/amount tokens.
    5. Unique token consumption: each PDF token `(page, token_idx)` can be claimed at most once.
    6. Exact-cent header/subtotal/grand-total arithmetic reconciliation.
    """
    if result.checksum_passed or provider is None:
        return result

    proposal = provider.propose_repair(result, tokens_by_page)
    if proposal is None:
        return result

    schema_err = _validate_schema(proposal)
    if schema_err is not None:
        return replace(result, fallback_rejected_reason=schema_err)

    src = f"llm_fallback_{provider.name}"
    ordered_by_page = index_tokens_by_page(tokens_by_page)
    used_tokens: set[TokenId] = set()

    inv_no_str = normalize_text(str(proposal["invoice_no"]))
    inv_no_prov = _ground_header_text(
        inv_no_str, ordered_by_page, used_tokens=used_tokens, source=src
    )
    if inv_no_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_invoice_no: {inv_no_str}",
        )

    inv_date_str = normalize_text(str(proposal["invoice_date"]))
    inv_date_prov = _ground_header_text(
        inv_date_str, ordered_by_page, used_tokens=used_tokens, source=src
    )
    if inv_date_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_invoice_date: {inv_date_str}",
        )

    subtotal_cents = int(proposal["subtotal_cents"])
    tax_cents = int(proposal["tax_cents"])
    credits_cents = int(proposal["credits_cents"])
    total_cents = int(proposal["total_cents"])

    total_prov = _ground_summary_cents(
        total_cents, ordered_by_page, used_tokens=used_tokens, source=src
    )
    if total_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_total_cents: {total_cents}",
        )

    tax_prov = _ground_summary_cents(
        tax_cents, ordered_by_page, used_tokens=used_tokens, source=src
    )
    if tax_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_tax_cents: {tax_cents}",
        )

    credits_prov: Provenance | None = None
    if credits_cents != 0:
        credits_prov = _ground_summary_cents(
            -abs(credits_cents), ordered_by_page, used_tokens=used_tokens, source=src
        ) or _ground_summary_cents(
            abs(credits_cents), ordered_by_page, used_tokens=used_tokens, source=src
        )
        if credits_prov is None:
            return replace(
                result,
                fallback_rejected_reason=f"ungrounded_credits_cents: {credits_cents}",
            )

    sub_prov = _ground_summary_cents(
        subtotal_cents, ordered_by_page, used_tokens=used_tokens, source=src
    )
    if sub_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_subtotal_cents: {subtotal_cents}",
        )

    repaired_items: list[LineItem] = []
    for idx, raw_item in enumerate(proposal["line_items"]):
        sku_str = normalize_text(str(raw_item["sku"]))
        desc_str = normalize_text(str(raw_item["description"]))
        qty_val = Decimal(str(raw_item["qty"]))
        unit_cents = int(raw_item["unit_price_cents"])
        amt_cents = int(raw_item["amount_cents"])

        if not verify_line_item_arithmetic(qty_val, unit_cents, amt_cents):
            return replace(
                result,
                fallback_rejected_reason=(
                    f"line_arithmetic_failed: line_items[{idx}] "
                    f"qty={qty_val} * unit_price_cents={unit_cents} != amount_cents={amt_cents}"
                ),
            )

        grounded_item, outcome = _ground_line_item_row(
            idx=idx,
            sku_str=sku_str,
            desc_str=desc_str,
            qty_val=qty_val,
            unit_cents=unit_cents,
            amt_cents=amt_cents,
            ordered_by_page=ordered_by_page,
            used_tokens=used_tokens,
            source=src,
            row_tol=row_tol,
        )
        if grounded_item is None:
            assert isinstance(outcome, str)
            return replace(result, fallback_rejected_reason=outcome)
        assert isinstance(outcome, set)
        used_tokens.update(outcome)
        repaired_items.append(grounded_item)

    line_sum = sum(it.amount_cents.value for it in repaired_items)
    if line_sum != subtotal_cents or (subtotal_cents + tax_cents - credits_cents) != total_cents:
        return replace(
            result,
            fallback_rejected_reason=(
                f"reconciliation_failed: line_sum={line_sum}, subtotal={subtotal_cents}, "
                f"tax={tax_cents}, credits={credits_cents}, total={total_cents}"
            ),
        )

    detail = (
        f"line_sum={line_sum} == subtotal={subtotal_cents}; "
        f"subtotal+tax-credits={subtotal_cents + tax_cents - credits_cents} == total={total_cents} "
        f"(repaired via {provider.name})"
    )
    return InvoiceResult(
        invoice_no=GroundedField(inv_no_str, inv_no_prov, confidence=0.9),
        invoice_date=GroundedField(inv_date_str, inv_date_prov, confidence=0.9),
        subtotal_cents=GroundedField(subtotal_cents, sub_prov, confidence=0.9),
        tax_cents=GroundedField(tax_cents, tax_prov, confidence=0.9),
        credits_cents=GroundedField(credits_cents, credits_prov, confidence=0.9),
        total_cents=GroundedField(total_cents, total_prov, confidence=0.9),
        line_items=repaired_items,
        checksum_passed=True,
        checksum_detail=detail,
        vendor_name=result.vendor_name,
        currency=result.currency,
        layout_hint=result.layout_hint,
        fallback_used=True,
        fallback_rejected_reason=None,
        latency_ms=result.latency_ms,
        pages=result.pages,
        warnings=[*result.warnings, f"repaired via strict-gated fallback ({provider.name})"],
    )
