from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any, Generic, TypeVar

T = TypeVar("T")

BBox = tuple[float, float, float, float]

LAYOUT_TYPES: tuple[str, ...] = (
    "standard_grid",
    "wrapped_header",
    "multipage_table",
    "credits_discounts",
    "group_subtotals",
    "jittered_ocr_layout",
)


def union_bbox(boxes: Sequence[BBox]) -> BBox:
    """Return the enclosing bounding box for a non-empty sequence of boxes."""
    if not boxes:
        raise ValueError("Cannot compute union_bbox of an empty sequence.")
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))


def decimal_to_json(value: Decimal) -> int | float:
    """Serialize a Decimal cleanly for JSON outputs."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return decimal_to_json(value)
    return value


@dataclass(frozen=True)
class BBoxToken:
    """Single word token with page-relative bounding box coordinates in PDF points.

    Coordinates follow pdfplumber convention: origin at top-left of each page
    (`y0` = top, `y1` = bottom, 1-indexed `page`).
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def bbox(self) -> BBox:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


@dataclass(frozen=True)
class Provenance:
    """Verifiable source span linking an extracted value back to PDF coordinates."""

    page: int
    bbox: BBox
    raw_text: str
    source: str

    @property
    def method(self) -> str:
        """Alias for `source` matching grounding contract terminology."""
        return self.source

    def is_valid(self) -> bool:
        x0, y0, x1, y1 = self.bbox
        return (
            self.page >= 1
            and bool(self.raw_text.strip())
            and bool(self.source.strip())
            and all(math.isfinite(v) for v in self.bbox)
            and x1 > x0
            and y1 > y0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "bbox": [round(v, 3) for v in self.bbox],
            "raw_text": self.raw_text,
            "source": self.source,
        }


@dataclass(frozen=True)
class GroundedField(Generic[T]):
    """Extracted scalar field accompanied by optional token provenance."""

    value: T
    provenance: Provenance | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _serialize_value(self.value),
            "confidence": round(self.confidence, 4),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass(frozen=True)
class LineItem:
    """Single invoice line item with field-level grounded coordinates and exact cents."""

    sku: GroundedField[str]
    description: GroundedField[str]
    qty: GroundedField[Decimal]
    unit_price_cents: GroundedField[int]
    amount_cents: GroundedField[int]
    page: int = 1
    row_bbox: BBox | None = None
    is_credit: bool = False

    def field_provenances(self) -> dict[str, Provenance | None]:
        return {
            "sku": self.sku.provenance,
            "description": self.description.provenance,
            "qty": self.qty.provenance,
            "unit_price_cents": self.unit_price_cents.provenance,
            "amount_cents": self.amount_cents.provenance,
        }

    def all_grounded(self) -> bool:
        return all(
            prov is not None and prov.is_valid() for prov in self.field_provenances().values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku.to_dict(),
            "description": self.description.to_dict(),
            "qty": self.qty.to_dict(),
            "unit_price_cents": self.unit_price_cents.to_dict(),
            "amount_cents": self.amount_cents.to_dict(),
            "page": self.page,
            "row_bbox": [round(v, 3) for v in self.row_bbox] if self.row_bbox else None,
            "is_credit": self.is_credit,
        }


@dataclass
class InvoiceResult:
    """Complete structured invoice extraction result with arithmetic audit trail."""

    invoice_no: GroundedField[str]
    invoice_date: GroundedField[str]
    subtotal_cents: GroundedField[int]
    tax_cents: GroundedField[int]
    credits_cents: GroundedField[int]
    total_cents: GroundedField[int]
    line_items: list[LineItem]
    checksum_passed: bool
    checksum_detail: str
    vendor_name: GroundedField[str] | None = None
    currency: str = "USD"
    layout_hint: str | None = None
    fallback_used: bool = False
    fallback_rejected_reason: str | None = None
    latency_ms: float = 0.0
    pages: int = 1
    warnings: list[str] = field(default_factory=list)

    @property
    def computed_line_sum_cents(self) -> int:
        return sum(item.amount_cents.value for item in self.line_items)

    def header_provenances(self) -> dict[str, Provenance | None]:
        return {
            "invoice_no": self.invoice_no.provenance,
            "invoice_date": self.invoice_date.provenance,
            "subtotal_cents": self.subtotal_cents.provenance,
            "tax_cents": self.tax_cents.provenance,
            "total_cents": self.total_cents.provenance,
        }

    def all_fields_grounded(self) -> bool:
        headers_ok = all(
            prov is not None and prov.is_valid() for prov in self.header_provenances().values()
        )
        items_ok = bool(self.line_items) and all(item.all_grounded() for item in self.line_items)
        return headers_ok and items_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_no": self.invoice_no.to_dict(),
            "invoice_date": self.invoice_date.to_dict(),
            "vendor_name": self.vendor_name.to_dict() if self.vendor_name else None,
            "currency": self.currency,
            "subtotal_cents": self.subtotal_cents.to_dict(),
            "tax_cents": self.tax_cents.to_dict(),
            "credits_cents": self.credits_cents.to_dict(),
            "total_cents": self.total_cents.to_dict(),
            "computed_line_sum_cents": self.computed_line_sum_cents,
            "checksum_passed": self.checksum_passed,
            "checksum_detail": self.checksum_detail,
            "layout_hint": self.layout_hint,
            "fallback_used": self.fallback_used,
            "fallback_rejected_reason": self.fallback_rejected_reason,
            "latency_ms": round(self.latency_ms, 3),
            "pages": self.pages,
            "warnings": list(self.warnings),
            "line_items": [item.to_dict() for item in self.line_items],
        }


@dataclass(frozen=True)
class GroundTruthLineItem:
    """Exact ground-truth line item emitted alongside synthetic PDFs."""

    sku: str
    description: str
    qty: Decimal
    unit_price_cents: int
    amount_cents: int
    is_credit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "description": self.description,
            "qty": decimal_to_json(self.qty),
            "unit_price_cents": self.unit_price_cents,
            "amount_cents": self.amount_cents,
            "is_credit": self.is_credit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthLineItem:
        return cls(
            sku=str(data["sku"]),
            description=str(data["description"]),
            qty=Decimal(str(data["qty"])),
            unit_price_cents=int(data["unit_price_cents"]),
            amount_cents=int(data["amount_cents"]),
            is_credit=bool(data.get("is_credit", False)),
        )


@dataclass(frozen=True)
class GroundTruthInvoice:
    """Exact ground-truth invoice metadata and line items for benchmark validation."""

    invoice_id: str
    layout: str
    vendor_name: str
    vendor_category: str
    invoice_no: str
    invoice_date: str
    currency: str
    line_items: tuple[GroundTruthLineItem, ...]
    subtotal_cents: int
    tax_cents: int
    credits_cents: int
    total_cents: int
    pdf_path: str
    group_subtotals_cents: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "layout": self.layout,
            "vendor_name": self.vendor_name,
            "vendor_category": self.vendor_category,
            "invoice_no": self.invoice_no,
            "invoice_date": self.invoice_date,
            "currency": self.currency,
            "line_items": [item.to_dict() for item in self.line_items],
            "subtotal_cents": self.subtotal_cents,
            "tax_cents": self.tax_cents,
            "credits_cents": self.credits_cents,
            "total_cents": self.total_cents,
            "pdf_path": self.pdf_path,
            "group_subtotals_cents": list(self.group_subtotals_cents),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthInvoice:
        return cls(
            invoice_id=str(data["invoice_id"]),
            layout=str(data["layout"]),
            vendor_name=str(data["vendor_name"]),
            vendor_category=str(data["vendor_category"]),
            invoice_no=str(data["invoice_no"]),
            invoice_date=str(data["invoice_date"]),
            currency=str(data.get("currency", "USD")),
            line_items=tuple(GroundTruthLineItem.from_dict(item) for item in data["line_items"]),
            subtotal_cents=int(data["subtotal_cents"]),
            tax_cents=int(data["tax_cents"]),
            credits_cents=int(data["credits_cents"]),
            total_cents=int(data["total_cents"]),
            pdf_path=str(data["pdf_path"]),
            group_subtotals_cents=tuple(int(x) for x in data.get("group_subtotals_cents", [])),
        )


@dataclass(frozen=True)
class LayoutMetrics:
    """Per-layout aggregate evaluation metrics."""

    layout: str
    count: int
    line_item_precision: float
    line_item_recall: float
    line_item_f1: float
    header_precision: float
    header_recall: float
    header_f1: float
    overall_field_f1: float
    checksum_pass_rate: float
    provenance_validity_rate: float
    p50_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate benchmark report across all evaluated invoices."""

    total_invoices: int
    seed: int
    line_item_precision: float
    line_item_recall: float
    line_item_f1: float
    header_precision: float
    header_recall: float
    header_f1: float
    overall_field_precision: float
    overall_field_recall: float
    overall_field_f1: float
    checksum_pass_rate: float
    exact_total_pass_count: int
    provenance_validity_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    per_layout: dict[str, LayoutMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_invoices": self.total_invoices,
            "seed": self.seed,
            "line_item_precision": round(self.line_item_precision, 6),
            "line_item_recall": round(self.line_item_recall, 6),
            "line_item_f1": round(self.line_item_f1, 6),
            "header_precision": round(self.header_precision, 6),
            "header_recall": round(self.header_recall, 6),
            "header_f1": round(self.header_f1, 6),
            "overall_field_precision": round(self.overall_field_precision, 6),
            "overall_field_recall": round(self.overall_field_recall, 6),
            "overall_field_f1": round(self.overall_field_f1, 6),
            "checksum_pass_rate": round(self.checksum_pass_rate, 6),
            "exact_total_pass_count": self.exact_total_pass_count,
            "provenance_validity_rate": round(self.provenance_validity_rate, 6),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "per_layout": {k: v.to_dict() for k, v in self.per_layout.items()},
        }
