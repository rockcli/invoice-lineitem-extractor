"""Strict-gated, token-grounded LLM fallback interface for checksum failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from invoice_extractor.grounding import (
    find_cents_token,
    find_qty_token,
    find_text_token_span,
    flatten_reading_order,
    normalize_text,
)
from invoice_extractor.models import (
    BBoxToken,
    GroundedField,
    InvoiceResult,
    LineItem,
    Provenance,
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


def _ground_description_tokens(
    desc_text: str,
    tokens_by_page: dict[int, list[BBoxToken]],
    *,
    source: str,
) -> Provenance | None:
    direct = find_text_token_span(desc_text, tokens_by_page, source=source)
    if direct is not None:
        return direct
    words = normalize_text(desc_text).split()
    if not words:
        return None
    for page in sorted(tokens_by_page):
        page_toks = tokens_by_page[page]
        matched: list[BBoxToken] = []
        cursor = 0
        for tok in flatten_reading_order(page_toks):
            if cursor < len(words) and normalize_text(tok.text) == words[cursor]:
                matched.append(tok)
                cursor += 1
        if cursor == len(words) and matched:
            x0 = min(t.x0 for t in matched)
            y0 = min(t.y0 for t in matched)
            x1 = max(t.x1 for t in matched)
            y1 = max(t.y1 for t in matched)
            return Provenance(
                page=page,
                bbox=(round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)),
                raw_text=" ".join(t.text for t in matched),
                source=source,
            )
    return None


def apply_strict_fallback(
    result: InvoiceResult,
    tokens_by_page: dict[int, list[BBoxToken]],
    provider: FallbackProvider | None,
) -> InvoiceResult:
    """Apply strict-gated fallback ONLY when `result.checksum_passed` is False.

    Enforces:
    1. Gate: Never invokes provider if deterministic checksum already passed.
    2. Strict JSON schema verification.
    3. Token-grounding verification against actual PDF tokens for every value.
    4. Exact-cent arithmetic reconciliation on the repaired structure.
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

    inv_no_str = normalize_text(str(proposal["invoice_no"]))
    inv_no_prov = find_text_token_span(inv_no_str, tokens_by_page, source=src)
    if inv_no_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_invoice_no: {inv_no_str}",
        )

    inv_date_str = normalize_text(str(proposal["invoice_date"]))
    inv_date_prov = find_text_token_span(inv_date_str, tokens_by_page, source=src)
    if inv_date_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_invoice_date: {inv_date_str}",
        )

    subtotal_cents = int(proposal["subtotal_cents"])
    tax_cents = int(proposal["tax_cents"])
    credits_cents = int(proposal["credits_cents"])
    total_cents = int(proposal["total_cents"])

    sub_prov = find_cents_token(subtotal_cents, tokens_by_page, source=src)
    if sub_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_subtotal_cents: {subtotal_cents}",
        )

    tax_prov = find_cents_token(tax_cents, tokens_by_page, source=src)
    if tax_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_tax_cents: {tax_cents}",
        )

    total_prov = find_cents_token(total_cents, tokens_by_page, source=src)
    if total_prov is None:
        return replace(
            result,
            fallback_rejected_reason=f"ungrounded_total_cents: {total_cents}",
        )

    credits_prov: Provenance | None = None
    if credits_cents != 0:
        credits_prov = find_cents_token(
            -abs(credits_cents), tokens_by_page, source=src
        ) or find_cents_token(abs(credits_cents), tokens_by_page, source=src)
        if credits_prov is None:
            return replace(
                result,
                fallback_rejected_reason=f"ungrounded_credits_cents: {credits_cents}",
            )

    repaired_items: list[LineItem] = []
    for idx, raw_item in enumerate(proposal["line_items"]):
        sku_str = normalize_text(str(raw_item["sku"]))
        desc_str = normalize_text(str(raw_item["description"]))
        qty_val = Decimal(str(raw_item["qty"]))
        unit_cents = int(raw_item["unit_price_cents"])
        amt_cents = int(raw_item["amount_cents"])

        sku_prov = find_text_token_span(sku_str, tokens_by_page, source=src)
        if sku_prov is None:
            return replace(
                result,
                fallback_rejected_reason=f"ungrounded_line_items[{idx}].sku: {sku_str}",
            )

        desc_prov = _ground_description_tokens(desc_str, tokens_by_page, source=src)
        if desc_prov is None:
            return replace(
                result,
                fallback_rejected_reason=f"ungrounded_line_items[{idx}].description: {desc_str}",
            )

        qty_prov = find_qty_token(qty_val, tokens_by_page, source=src)
        if qty_prov is None:
            return replace(
                result,
                fallback_rejected_reason=f"ungrounded_line_items[{idx}].qty: {qty_val}",
            )

        unit_prov = find_cents_token(unit_cents, tokens_by_page, source=src)
        if unit_prov is None:
            return replace(
                result,
                fallback_rejected_reason=(
                    f"ungrounded_line_items[{idx}].unit_price_cents: {unit_cents}"
                ),
            )

        amt_prov = find_cents_token(amt_cents, tokens_by_page, source=src)
        if amt_prov is None:
            return replace(
                result,
                fallback_rejected_reason=f"ungrounded_line_items[{idx}].amount_cents: {amt_cents}",
            )

        repaired_items.append(
            LineItem(
                sku=GroundedField(sku_str, sku_prov, confidence=0.9),
                description=GroundedField(desc_str, desc_prov, confidence=0.9),
                qty=GroundedField(qty_val, qty_prov, confidence=0.9),
                unit_price_cents=GroundedField(unit_cents, unit_prov, confidence=0.9),
                amount_cents=GroundedField(amt_cents, amt_prov, confidence=0.9),
                page=amt_prov.page,
                is_credit=amt_cents < 0,
            )
        )

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
