"""Benchmark evaluation harness and real-data visual asset generator."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from invoice_extractor.generator import PAGE_HEIGHT, PAGE_WIDTH, format_cents, generate_dataset
from invoice_extractor.grounding import verify_invoice_grounding
from invoice_extractor.models import (
    LAYOUT_TYPES,
    EvalSummary,
    GroundTruthInvoice,
    GroundTruthLineItem,
    InvoiceResult,
    LayoutMetrics,
    LineItem,
)
from invoice_extractor.parser import extract_tokens, parse_invoice


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: _Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _score_header_fields(result: InvoiceResult, gt: GroundTruthInvoice) -> _Counts:
    counts = _Counts()
    pairs: list[tuple[Any, Any]] = [
        (result.invoice_no.value, gt.invoice_no),
        (result.invoice_date.value, gt.invoice_date),
        (result.subtotal_cents.value, gt.subtotal_cents),
        (result.tax_cents.value, gt.tax_cents),
        (result.credits_cents.value, gt.credits_cents),
        (result.total_cents.value, gt.total_cents),
    ]
    for pred_val, gt_val in pairs:
        if pred_val == gt_val:
            counts.tp += 1
        else:
            if pred_val not in ("", None):
                counts.fp += 1
            counts.fn += 1
    return counts


def _score_line_items(
    pred_items: list[LineItem],
    gt_items: tuple[GroundTruthLineItem, ...],
) -> _Counts:
    """Align predicted items against ground truth items by SKU/order and score 5 fields each."""
    counts = _Counts()
    used_gt: set[int] = set()

    for pred in pred_items:
        match_idx: int | None = None
        for idx, gt_it in enumerate(gt_items):
            if idx not in used_gt and pred.sku.value == gt_it.sku:
                match_idx = idx
                break
        if match_idx is None:
            for idx, gt_it in enumerate(gt_items):
                if idx not in used_gt and pred.amount_cents.value == gt_it.amount_cents:
                    match_idx = idx
                    break
        if match_idx is None:
            counts.fp += 5
            continue

        used_gt.add(match_idx)
        gt_it = gt_items[match_idx]
        field_pairs: list[tuple[Any, Any]] = [
            (pred.sku.value, gt_it.sku),
            (pred.description.value, gt_it.description),
            (pred.qty.value, gt_it.qty),
            (pred.unit_price_cents.value, gt_it.unit_price_cents),
            (pred.amount_cents.value, gt_it.amount_cents),
        ]
        for p_val, g_val in field_pairs:
            if p_val == g_val:
                counts.tp += 1
            else:
                counts.fp += 1
                counts.fn += 1

    unmatched_gt = len(gt_items) - len(used_gt)
    counts.fn += unmatched_gt * 5
    return counts


def _render_annotated_invoice_image(
    pdf_path: Path,
    result: InvoiceResult,
    *,
    scale: float = 2.2,
) -> Image.Image:
    """Render page 1 of a real PDF and draw color-coded Provenance bounding boxes."""
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    bitmap = page.render(scale=scale)
    pil_img = bitmap.to_pil()
    assert isinstance(pil_img, Image.Image)
    img: Image.Image = pil_img.convert("RGB")
    doc.close()

    draw = ImageDraw.Draw(img, "RGBA")
    sx = img.width / PAGE_WIDTH
    sy = img.height / PAGE_HEIGHT

    def to_px(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        return (x0 * sx - 2, y0 * sy - 2, x1 * sx + 2, y1 * sy + 2)

    # Header grounded fields in deep blue
    for prov in result.header_provenances().values():
        if prov is not None and prov.page == 1:
            box = to_px(prov.bbox)
            draw.rectangle(box, fill=(37, 99, 235, 34), outline=(29, 78, 216, 230), width=2)

    # Line items in emerald green (or amber for credits)
    for item in result.line_items:
        if item.page != 1:
            continue
        fill = (217, 119, 6, 36) if item.is_credit else (16, 185, 129, 32)
        outline = (180, 83, 9, 230) if item.is_credit else (5, 150, 105, 230)
        for prov in item.field_provenances().values():
            if prov is not None and prov.page == 1:
                draw.rectangle(to_px(prov.bbox), fill=fill, outline=outline, width=2)

    return img


def generate_demo_overview_asset(
    sample_pdf: Path,
    sample_result: InvoiceResult,
    summary: EvalSummary,
    output_path: Path,
) -> Path:
    """Create assets/demo_overview.png from a real parsed PDF and benchmark metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = _render_annotated_invoice_image(sample_pdf, sample_result, scale=2.2)

    fig = plt.figure(figsize=(16.0, 9.2), dpi=140, facecolor="#F8FAFC")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.15],
        height_ratios=[1.0, 1.0],
        wspace=0.16,
        hspace=0.34,
    )

    # Left: real annotated PDF
    ax_pdf = fig.add_subplot(gs[:, 0])
    ax_pdf.imshow(annotated)
    ax_pdf.set_xticks([])
    ax_pdf.set_yticks([])
    for spine in ax_pdf.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.4)
    ax_pdf.set_title(
        f"Real Grounded Parse: {sample_pdf.name}\n"
        f"Blue = Header Totals/IDs | Green = Line Items | Subtotals Excluded",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )

    # Top-right: Per-layout F1 & Checksum Pass Rate
    ax_f1 = fig.add_subplot(gs[0, 1])
    layouts = list(LAYOUT_TYPES)
    short_labels = [
        "Standard\nGrid",
        "Wrapped\nHeader",
        "Multi-Page\nTable",
        "Credits &\nDiscounts",
        "Group\nSubtotals",
        "Jittered\nOCR ±2.5pt",
    ]
    f1_vals = [summary.per_layout[k].overall_field_f1 * 100.0 for k in layouts]
    chk_vals = [summary.per_layout[k].checksum_pass_rate * 100.0 for k in layouts]

    x_pos = list(range(len(layouts)))
    width = 0.36
    bars1 = ax_f1.bar(
        [x - width / 2 for x in x_pos],
        f1_vals,
        width=width,
        color="#2563EB",
        label="Field F1 (%)",
    )
    bars2 = ax_f1.bar(
        [x + width / 2 for x in x_pos],
        chk_vals,
        width=width,
        color="#059669",
        label="Exact Checksum (%)",
    )
    ax_f1.set_ylim(0, 124)
    ax_f1.set_xticks(x_pos)
    ax_f1.set_xticklabels(short_labels, fontsize=8.8, color="#1E293B")
    ax_f1.set_ylabel("Percentage (%)", fontsize=9.5, color="#334155")
    ax_f1.set_title(
        f"120-Invoice Benchmark ({summary.total_invoices} PDFs, seed={summary.seed}) — "
        f"Field F1 {summary.overall_field_f1 * 100:.1f}% | "
        f"Exact Total {summary.checksum_pass_rate * 100:.1f}%",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_f1.grid(axis="y", linestyle="--", alpha=0.4)
    ax_f1.spines["top"].set_visible(False)
    ax_f1.spines["right"].set_visible(False)
    ax_f1.legend(loc="upper right", ncol=2, frameon=True, fontsize=8.5)
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.5,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#1E293B",
            fontweight="bold",
        )

    # Bottom-right: Per-layout Latency (p50 vs p95 ms)
    ax_lat = fig.add_subplot(gs[1, 1])
    p50_vals = [summary.per_layout[k].p50_latency_ms for k in layouts]
    p95_vals = [summary.per_layout[k].p95_latency_ms for k in layouts]
    b_p50 = ax_lat.bar(
        [x - width / 2 for x in x_pos],
        p50_vals,
        width=width,
        color="#0EA5E9",
        label="p50 ms",
    )
    b_p95 = ax_lat.bar(
        [x + width / 2 for x in x_pos],
        p95_vals,
        width=width,
        color="#F59E0B",
        label="p95 ms",
    )
    max_lat = max([*p95_vals, 10.0]) * 1.25
    ax_lat.set_ylim(0, max_lat)
    ax_lat.set_xticks(x_pos)
    ax_lat.set_xticklabels(short_labels, fontsize=8.8, color="#1E293B")
    ax_lat.set_ylabel("Latency (ms / PDF)", fontsize=9.5, color="#334155")
    ax_lat.set_title(
        f"Per-Invoice Parse Latency — Overall p50: {summary.p50_latency_ms:.1f} ms | "
        f"p95: {summary.p95_latency_ms:.1f} ms | Provenance Valid: "
        f"{summary.provenance_validity_rate * 100:.1f}%",
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax_lat.grid(axis="y", linestyle="--", alpha=0.4)
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)
    ax_lat.legend(loc="upper left", frameon=True, fontsize=8.5)
    for bar in [*b_p50, *b_p95]:
        h = bar.get_height()
        ax_lat.text(
            bar.get_x() + bar.get_width() / 2,
            h + max_lat * 0.015,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#1E293B",
        )

    fig.suptitle(
        "Invoice Line-Item Extractor — Reference Implementation (Synthetic Data)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.89, bottom=0.07)
    fig.savefig(output_path, dpi=140, facecolor="#F8FAFC")
    plt.close(fig)
    return output_path


def generate_cover_asset(
    sample_pdf: Path,
    sample_result: InvoiceResult,
    summary: EvalSummary,
    output_path: Path,
) -> Path:
    """Create exact 1600x1200 cover image from real evaluation results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = _render_annotated_invoice_image(sample_pdf, sample_result, scale=2.3)

    fig = plt.figure(figsize=(16.0, 12.0), dpi=100, facecolor="#0F172A")

    # Top title banner axes
    ax_banner = fig.add_axes((0.03, 0.87, 0.94, 0.105), facecolor="#0F172A")
    ax_banner.axis("off")
    ax_banner.text(
        0.0,
        0.68,
        "Invoice Line-Item Extractor",
        fontsize=27,
        fontweight="bold",
        color="#F8FAFC",
    )
    ax_banner.text(
        0.0,
        0.18,
        "Geometry-First PDF Extraction · Exact-Cent Reconciliation · Token-BBox Provenance",
        fontsize=12.8,
        color="#94A3B8",
    )
    badge = patches.FancyBboxPatch(
        (0.645, 0.48),
        0.33,
        0.38,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor="#1E293B",
        edgecolor="#38BDF8",
        linewidth=1.5,
    )
    ax_banner.add_patch(badge)
    ax_banner.text(
        0.81,
        0.67,
        "Reference implementation · synthetic data",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#38BDF8",
        parse_math=False,
    )

    # Left card: Annotated PDF with Provenance overlays
    ax_pdf = fig.add_axes((0.035, 0.05, 0.455, 0.79), facecolor="#FFFFFF")
    ax_pdf.imshow(annotated)
    ax_pdf.set_xticks([])
    ax_pdf.set_yticks([])
    for spine in ax_pdf.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(2.0)
    ax_pdf.set_title(
        f"Verified BBox Grounding ({sample_pdf.name})",
        fontsize=12.5,
        fontweight="bold",
        color="#E2E8F0",
        pad=10,
    )

    # Right top: 4 KPI tiles
    kpi_specs = [
        ("SYNTHETIC SUITE", f"{summary.total_invoices} PDFs", "6 Vendor Layouts", "#38BDF8"),
        (
            "OVERALL FIELD F1",
            f"{summary.overall_field_f1 * 100:.1f}%",
            (
                f"P {summary.overall_field_precision * 100:.1f}% · "
                f"R {summary.overall_field_recall * 100:.1f}%"
            ),
            "#34D399",
        ),
        (
            "EXACT CHECKSUM",
            f"{summary.exact_total_pass_count}/{summary.total_invoices}",
            f"{summary.provenance_validity_rate * 100:.1f}% BBox Valid",
            "#A78BFA",
        ),
        (
            "PARSE LATENCY",
            f"{summary.p50_latency_ms:.1f} ms",
            f"p95 {summary.p95_latency_ms:.1f} ms / PDF",
            "#FBBF24",
        ),
    ]
    for idx, (label, big_val, sub_val, accent) in enumerate(kpi_specs):
        col = idx % 2
        row = idx // 2
        x0 = 0.52 + col * 0.232
        y0 = 0.695 - row * 0.145
        ax_kpi = fig.add_axes((x0, y0, 0.215, 0.125), facecolor="#1E293B")
        ax_kpi.set_xticks([])
        ax_kpi.set_yticks([])
        for spine in ax_kpi.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.4)
        ax_kpi.text(0.08, 0.76, label, fontsize=9.2, fontweight="bold", color="#94A3B8")
        ax_kpi.text(0.08, 0.40, big_val, fontsize=21, fontweight="bold", color=accent)
        ax_kpi.text(0.08, 0.14, sub_val, fontsize=9.2, color="#CBD5E1")

    # Right middle: Per-layout F1 bar chart
    ax_chart = fig.add_axes((0.545, 0.285, 0.42, 0.235), facecolor="#1E293B")
    layouts = list(LAYOUT_TYPES)
    names = [
        "Standard",
        "Wrapped\nHdr",
        "Multi-\nPage",
        "Credits",
        "Group\nSubtotals",
        "Jittered\nOCR",
    ]
    vals = [summary.per_layout[k].overall_field_f1 * 100.0 for k in layouts]
    bars = ax_chart.bar(range(len(layouts)), vals, color="#10B981", width=0.55)
    ax_chart.set_ylim(0, 118)
    ax_chart.set_xticks(range(len(layouts)))
    ax_chart.set_xticklabels(names, fontsize=8.8, color="#E2E8F0")
    ax_chart.tick_params(axis="y", colors="#94A3B8", labelsize=8.5)
    ax_chart.set_title(
        "Field F1 (%) by Vendor Layout",
        fontsize=11,
        fontweight="bold",
        color="#F8FAFC",
        pad=8,
    )
    ax_chart.grid(axis="y", linestyle="--", color="#334155", alpha=0.8)
    ax_chart.spines["top"].set_visible(False)
    ax_chart.spines["right"].set_visible(False)
    ax_chart.spines["left"].set_color("#475569")
    ax_chart.spines["bottom"].set_color("#475569")
    for bar in bars:
        h = bar.get_height()
        ax_chart.text(
            bar.get_x() + bar.get_width() / 2,
            h + 2.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#34D399",
        )

    # Right bottom: Exact-cent audit card from sample_result
    ax_audit = fig.add_axes((0.52, 0.05, 0.445, 0.185), facecolor="#1E293B")
    ax_audit.set_xticks([])
    ax_audit.set_yticks([])
    for spine in ax_audit.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.4)
    line_sum_str = format_cents(sample_result.computed_line_sum_cents)
    sub_str = format_cents(sample_result.subtotal_cents.value)
    tax_str = format_cents(sample_result.tax_cents.value)
    tot_str = format_cents(sample_result.total_cents.value)
    audit_lines = (
        f"Exact-Cent Reconciliation Audit ({sample_result.invoice_no.value})\n"
        f"  Extracted Line Items: {len(sample_result.line_items)}  |  "
        f"Sum(Line Items): {line_sum_str} == Subtotal: {sub_str}\n"
        f"  Subtotal + Tax - Credits: {sub_str} + {tax_str} - "
        f"{format_cents(sample_result.credits_cents.value)} == Grand Total: {tot_str}\n"
        f"  Strict Fallback Gate: checksum_passed={sample_result.checksum_passed}  "
        f"fallback_used={sample_result.fallback_used}"
    )
    ax_audit.text(
        0.04,
        0.50,
        audit_lines,
        va="center",
        ha="left",
        fontsize=9.2,
        family="monospace",
        color="#E2E8F0",
        linespacing=1.6,
        parse_math=False,
    )

    fig.savefig(output_path, dpi=100, facecolor="#0F172A")
    plt.close(fig)

    # Guarantee exact 1600x1200 pixels
    with Image.open(output_path) as im:
        if im.size != (1600, 1200):
            resized = im.resize((1600, 1200), Image.Resampling.LANCZOS)
            resized.save(output_path)

    return output_path


def run_evaluation(
    *,
    data_dir: Path,
    report_path: Path,
    assets_dir: Path | None = None,
    count: int = 120,
    seed: int = 42,
    generate_assets: bool = True,
) -> EvalSummary:
    """Generate/load `count` synthetic invoices, evaluate extraction, and save reports/assets."""
    ground_truths = generate_dataset(data_dir, count=count, seed=seed)

    total_hdr = _Counts()
    total_items = _Counts()
    latencies: list[float] = []
    checksum_passes = 0
    prov_total_fields = 0
    prov_valid_fields = 0

    per_layout_hdr: dict[str, _Counts] = defaultdict(_Counts)
    per_layout_items: dict[str, _Counts] = defaultdict(_Counts)
    per_layout_latencies: dict[str, list[float]] = defaultdict(list)
    per_layout_passes: dict[str, int] = defaultdict(int)
    per_layout_counts: dict[str, int] = defaultdict(int)
    per_layout_prov_total: dict[str, int] = defaultdict(int)
    per_layout_prov_valid: dict[str, int] = defaultdict(int)

    sample_pair: tuple[Path, InvoiceResult] | None = None

    for gt in ground_truths:
        pdf_path = data_dir / gt.pdf_path
        result = parse_invoice(pdf_path, layout_hint=gt.layout)
        tokens_by_page, _ = extract_tokens(pdf_path)
        audit = verify_invoice_grounding(result, tokens_by_page)

        hdr_c = _score_header_fields(result, gt)
        item_c = _score_line_items(result.line_items, gt.line_items)

        total_hdr.add(hdr_c)
        total_items.add(item_c)
        latencies.append(result.latency_ms)

        exact_ok = (
            result.checksum_passed
            and result.total_cents.value == gt.total_cents
            and result.subtotal_cents.value == gt.subtotal_cents
        )
        if exact_ok:
            checksum_passes += 1
            per_layout_passes[gt.layout] += 1

        prov_total_fields += audit.total_fields
        prov_valid_fields += audit.grounded_fields

        per_layout_hdr[gt.layout].add(hdr_c)
        per_layout_items[gt.layout].add(item_c)
        per_layout_latencies[gt.layout].append(result.latency_ms)
        per_layout_counts[gt.layout] += 1
        per_layout_prov_total[gt.layout] += audit.total_fields
        per_layout_prov_valid[gt.layout] += audit.grounded_fields

        if gt.layout == "group_subtotals" and sample_pair is None:
            sample_pair = (pdf_path, result)

    if sample_pair is None and ground_truths:
        first_pdf = data_dir / ground_truths[0].pdf_path
        sample_pair = (first_pdf, parse_invoice(first_pdf))

    overall_counts = _Counts(
        tp=total_hdr.tp + total_items.tp,
        fp=total_hdr.fp + total_items.fp,
        fn=total_hdr.fn + total_items.fn,
    )

    per_layout_metrics: dict[str, LayoutMetrics] = {}
    for layout in LAYOUT_TYPES:
        cnt = per_layout_counts[layout]
        if cnt == 0:
            continue
        l_hdr = per_layout_hdr[layout]
        l_itm = per_layout_items[layout]
        l_all = _Counts(tp=l_hdr.tp + l_itm.tp, fp=l_hdr.fp + l_itm.fp, fn=l_hdr.fn + l_itm.fn)
        p_tot = per_layout_prov_total[layout]
        per_layout_metrics[layout] = LayoutMetrics(
            layout=layout,
            count=cnt,
            line_item_precision=round(l_itm.precision(), 6),
            line_item_recall=round(l_itm.recall(), 6),
            line_item_f1=round(l_itm.f1(), 6),
            header_precision=round(l_hdr.precision(), 6),
            header_recall=round(l_hdr.recall(), 6),
            header_f1=round(l_hdr.f1(), 6),
            overall_field_f1=round(l_all.f1(), 6),
            checksum_pass_rate=round(per_layout_passes[layout] / cnt, 6),
            provenance_validity_rate=round(
                per_layout_prov_valid[layout] / p_tot if p_tot > 0 else 0.0, 6
            ),
            p50_latency_ms=round(_percentile(per_layout_latencies[layout], 50.0), 3),
            p95_latency_ms=round(_percentile(per_layout_latencies[layout], 95.0), 3),
        )

    summary = EvalSummary(
        total_invoices=len(ground_truths),
        seed=seed,
        line_item_precision=total_items.precision(),
        line_item_recall=total_items.recall(),
        line_item_f1=total_items.f1(),
        header_precision=total_hdr.precision(),
        header_recall=total_hdr.recall(),
        header_f1=total_hdr.f1(),
        overall_field_precision=overall_counts.precision(),
        overall_field_recall=overall_counts.recall(),
        overall_field_f1=overall_counts.f1(),
        checksum_pass_rate=checksum_passes / len(ground_truths) if ground_truths else 0.0,
        exact_total_pass_count=checksum_passes,
        provenance_validity_rate=(
            prov_valid_fields / prov_total_fields if prov_total_fields > 0 else 0.0
        ),
        p50_latency_ms=_percentile(latencies, 50.0),
        p95_latency_ms=_percentile(latencies, 95.0),
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        per_layout=per_layout_metrics,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")

    if generate_assets and assets_dir is not None and sample_pair is not None:
        sample_pdf, sample_result = sample_pair
        generate_demo_overview_asset(
            sample_pdf,
            sample_result,
            summary,
            assets_dir / "demo_overview.png",
        )
        generate_cover_asset(
            sample_pdf,
            sample_result,
            summary,
            assets_dir / "cover_1600x1200.png",
        )

    return summary
