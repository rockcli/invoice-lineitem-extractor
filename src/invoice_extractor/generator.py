"""Deterministic synthetic PDF invoice generator across 6 challenging vendor layouts."""

from __future__ import annotations

import json
import random
from decimal import Decimal
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen.canvas import Canvas

from invoice_extractor.models import (
    LAYOUT_TYPES,
    GroundTruthInvoice,
    GroundTruthLineItem,
)

PAGE_WIDTH, PAGE_HEIGHT = LETTER  # 612 x 792 pt
MARGIN_X = 46.0

_VENDOR_CATALOG: dict[str, dict[str, object]] = {
    "standard_grid": {
        "vendor": "Apex Industrial Hardware Co.",
        "category": "industrial_hardware",
        "address": "410 Commerce Parkway, Columbus, OH 43215",
        "tax_id": "31-4829104",
        "items": [
            ("HW-M12-HEX", "M12 Stainless Steel Hex Bolt Assembly (Box/50)"),
            ("HW-BRG-204", "Sealed Radial Ball Bearing Housing 20mm Bore"),
            ("HW-ACT-088", "Double-Acting Pneumatic Actuator Cylinder 40mm"),
            ("HW-FLG-316", "316L Stainless Weld-Neck Pipe Flange 2-Inch"),
            ("HW-GSK-410", "Spiral-Wound Graphite Ring Gasket Class 300"),
            ("HW-VLV-112", "Forged Brass Inline Check Valve 3/4 NPT"),
            ("HW-CLM-025", "Heavy-Duty T-Bolt Hose Clamp Set (Pack/10)"),
            ("HW-SHM-090", "Precision Alignment Steel Shim Kit 0.05-1.0mm"),
        ],
    },
    "wrapped_header": {
        "vendor": "Stratos Cloud Equipment Inc.",
        "category": "cloud_equipment",
        "address": "850 Meridian Blvd, Suite 400, Austin, TX 78758",
        "tax_id": "74-3910482",
        "items": [
            (
                "CE-RCK-42U",
                "42U Perforated Server Rack Cabinet with Dual Rear Cable Channels and Casters",
            ),
            (
                "CE-OPT-100",
                "100GBASE-LR4 QSFP28 Single-Mode Optical Transceiver Module 10km Reach",
            ),
            (
                "CE-PDU-30A",
                "Switched Metered Rack PDU 30A 208V 24-Outlet C13/C19 with Network Card",
            ),
            (
                "CE-NIC-25G",
                "Dual-Port 25GbE SFP28 PCIe 4.0 Network Interface Adapter Low-Profile Bracket",
            ),
            (
                "CE-UPS-3KV",
                "Online Double-Conversion 3000VA 2U Rackmount UPS with Extended Battery Port",
            ),
            (
                "CE-KVM-16P",
                "16-Port Cat5e IP Console Switch with Local VGA/USB Management Interface",
            ),
            (
                "CE-SAS-012",
                "12Gbps External Mini-SAS HD Shielded Interconnect Cable Assembly 2.0m",
            ),
        ],
    },
    "multipage_table": {
        "vendor": "Meridian Logistics Freight Ltd.",
        "category": "logistics_freight",
        "address": "1220 Distribution Way, Chicago, IL 60638",
        "tax_id": "36-7102945",
        "items": [
            ("FRT-LTL-101", "Regional LTL Palletized Linehaul Midwest Corridor"),
            ("FRT-LFT-202", "Origin Dock-High Liftgate Pickup Handling Charge"),
            ("FRT-FSC-303", "Weekly Indexed Highway Diesel Fuel Surcharge"),
            ("FRT-TMP-404", "Temperature-Controlled Reefer Monitoring Fee"),
            ("FRT-HAZ-505", "Class 9 Non-Bulk Hazardous Documentation Fee"),
            ("FRT-DET-606", "Carrier Terminal Detention Over 90 Minutes"),
            ("FRT-PAL-707", "ISPM-15 Heat-Treated Hardwood Pallet Exchange"),
            ("FRT-WRP-808", "Shrink-Wrap & Cornerboard Re-Palletization Service"),
            ("FRT-CUS-909", "Automated Customs Pre-Arrival Manifest Filing"),
            ("FRT-APT-110", "Scheduled Consignee Delivery Appointment Fee"),
            ("FRT-RES-111", "Commercial Limited-Access Gate Clearance Charge"),
            ("FRT-INS-112", "All-Risk Cargo Transit Valuation Coverage"),
            ("FRT-STO-113", "Bonded Cross-Dock Short-Term Storage Per Day"),
            ("FRT-OVS-114", "Over-Length Freight Handling Surcharge > 96in"),
            ("FRT-WGT-115", "Certified Axle Scale Re-Weigh Verification"),
            ("FRT-TRK-116", "GPS Seal Telemetry & Route Exception Alerting"),
            ("FRT-POD-117", "Signed Bill-of-Lading Archival & Scan Indexing"),
            ("FRT-DOC-118", "Export Commercial Invoice & Packing List Prep"),
        ],
    },
    "credits_discounts": {
        "vendor": "Vanguard Office Supply Corp.",
        "category": "office_supplies",
        "address": "305 Peachtree Industrial Rd, Charlotte, NC 28208",
        "tax_id": "56-8204193",
        "items": [
            ("OFF-CHR-210", "Ergonomic Mesh Task Chair Adjustable Lumbar Support"),
            ("OFF-DSK-315", "Electric Dual-Motor Sit-Stand Desk Frame 60x30"),
            ("OFF-PNL-108", "Acoustic Felt Desk Partition Panel 48-Inch Charcoal"),
            ("OFF-LBL-406", "Direct Thermal Shipping Labels 4x6 Roll (500ct)"),
            ("OFF-PPR-851", "24lb Bright Recycled Laser Copy Paper Case (10 Ream)"),
            ("OFF-ARM-220", "Pneumatic Dual-Monitor Articulating Desk Mount"),
            ("OFF-CAB-305", "3-Drawer Steel Rolling Pedestal Filing Cabinet"),
        ],
        "credits": [
            ("CR-RET-01", "Damaged Carton Return Credit Memo"),
            ("DISC-VOL-05", "Contract Tier Volume Rebate Adjustment"),
            ("CR-CORE-02", "Returned Packaging Tote Deposit Credit"),
        ],
    },
    "group_subtotals": {
        "vendor": "Helios Lab Consumables LLC",
        "category": "lab_consumables",
        "address": "740 Research Park Dr, Pittsburgh, PA 15213",
        "tax_id": "25-9041827",
        "sections": [
            (
                "Section A: Volumetric Glassware",
                [
                    ("LAB-FLK-500", "Class A Borosilicate Volumetric Flask 500mL"),
                    ("LAB-BKR-250", "Low-Form Graduated Glass Beaker 250mL (Pack/12)"),
                    ("LAB-CYL-100", "Hexagonal Base Graduated Cylinder 100mL"),
                ],
            ),
            (
                "Section B: Liquid Handling & Tips",
                [
                    ("LAB-TIP-200", "Sterile Low-Retention Filter Pipette Tips 200uL"),
                    ("LAB-TIP-1000", "DNase-Free Aerosol Barrier Tips 1000uL Rack"),
                    ("LAB-SRP-010", "Individually Wrapped Serological Pipette 10mL"),
                ],
            ),
            (
                "Section C: Cryogenic & Sample Storage",
                [
                    ("LAB-VIAL-02", "External-Thread Cryogenic Vial 2.0mL (Pack/100)"),
                    ("LAB-BOX-081", "Polycarbonate 81-Well Freezer Storage Box"),
                    ("LAB-LBL-CRY", "Cryo-Resistant Thermal Barcode Roll -80C Rated"),
                ],
            ),
        ],
    },
    "jittered_ocr_layout": {
        "vendor": "Beacon Electrical Components Co.",
        "category": "electrical_components",
        "address": "1950 Foundry Ave, Denver, CO 80216",
        "tax_id": "84-6219305",
        "items": [
            ("ELC-PSU-24V", "DIN Rail 24VDC 10A Industrial Switching Power Supply"),
            ("ELC-CBL-100", "Shielded Twisted-Pair Instrumentation Cable 100m"),
            ("ELC-BRK-060", "3-Pole Molded Case Circuit Breaker 60A 480VAC"),
            ("ELC-RLY-012", "Slimline Electromechanical Relay Module 24VDC"),
            ("ELC-TRM-035", "Spring-Clamp Modular Terminal Block Gray (Box/50)"),
            ("ELC-ENC-161", "NEMA 4X Fiberglass Wall-Mount Control Enclosure"),
            ("ELC-SPD-275", "Type 2 Surge Protective Device 275VAC Modular"),
        ],
    },
}


def format_cents(cents: int, *, parens_negative: bool = False) -> str:
    """Format integer cents into USD currency string."""
    abs_val = abs(cents)
    dollars = abs_val // 100
    rem = abs_val % 100
    formatted = f"${dollars:,}.{rem:02d}"
    if cents < 0:
        return f"({formatted})" if parens_negative else f"-{dollars:,}.{rem:02d}"
    return formatted


def _wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _build_ground_truth(
    index: int,
    layout: str,
    rng: random.Random,
    pdf_rel_path: str,
) -> tuple[GroundTruthInvoice, list[tuple[str, list[GroundTruthLineItem]]]]:
    catalog = _VENDOR_CATALOG[layout]
    vendor = str(catalog["vendor"])
    category = str(catalog["category"])
    day = 1 + (index * 3 + rng.randint(0, 2)) % 28
    month = 1 + (index + rng.randint(0, 1)) % 10
    invoice_date = f"2026-{month:02d}-{day:02d}"
    invoice_no = f"INV-2026-{1000 + index:04d}"

    sections: list[tuple[str, list[GroundTruthLineItem]]] = []
    all_items: list[GroundTruthLineItem] = []

    def make_item(sku: str, desc: str, *, negative: bool = False) -> GroundTruthLineItem:
        if negative:
            qty = Decimal("1")
            unit_cents = -rng.choice([2500, 3500, 4500, 6250, 8500, 12000])
        else:
            use_half = rng.random() < 0.25
            if use_half:
                qty = Decimal(str(rng.choice([1.5, 2.5, 3.5])))
                unit_cents = rng.randrange(1200, 24000, 200)
            else:
                qty = Decimal(str(rng.randint(1, 12)))
                unit_cents = rng.randrange(850, 32500, 25)
        amount_cents = int(qty * unit_cents)
        return GroundTruthLineItem(
            sku=sku,
            description=desc,
            qty=qty,
            unit_price_cents=unit_cents,
            amount_cents=amount_cents,
            is_credit=negative,
        )

    group_subtotals: list[int] = []

    if layout == "group_subtotals":
        raw_sections = catalog["sections"]
        assert isinstance(raw_sections, list)
        chosen_sections = raw_sections if rng.random() < 0.7 else raw_sections[:2]
        for sec_title, sec_pool in chosen_sections:
            k = rng.randint(2, len(sec_pool))
            picked = rng.sample(sec_pool, k=k)
            sec_items = [make_item(sku, desc) for sku, desc in picked]
            sections.append((sec_title, sec_items))
            all_items.extend(sec_items)
            group_subtotals.append(sum(it.amount_cents for it in sec_items))
    else:
        pool = catalog["items"]
        assert isinstance(pool, list)
        if layout == "multipage_table":
            count_items = rng.randint(15, 18)
            picked = rng.sample(pool, k=count_items)
        elif layout == "credits_discounts":
            count_items = rng.randint(4, 6)
            picked = rng.sample(pool, k=count_items)
        else:
            count_items = rng.randint(4, 7)
            picked = rng.sample(pool, k=min(count_items, len(pool)))

        items = [make_item(sku, desc) for sku, desc in picked]

        if layout == "credits_discounts":
            credit_pool = catalog["credits"]
            assert isinstance(credit_pool, list)
            num_credits = rng.choice([1, 2])
            for c_sku, c_desc in rng.sample(credit_pool, k=num_credits):
                items.append(make_item(c_sku, c_desc, negative=True))

        sections.append(("", items))
        all_items.extend(items)

    subtotal_cents = sum(it.amount_cents for it in all_items)
    # Ensure positive invoice subtotal even on credit layouts
    if subtotal_cents <= 5000:
        boost = make_item("ADJ-STD-99", "Standard Assembly Handling Fee")
        all_items.insert(0, boost)
        sections[0][1].insert(0, boost)
        subtotal_cents = sum(it.amount_cents for it in all_items)

    credits_cents = 0
    if layout == "credits_discounts" and (index % 2 == 0):
        credits_cents = rng.choice([1500, 2500, 4000])
        if credits_cents >= subtotal_cents:
            credits_cents = 1000

    tax_bps = rng.choice([0, 625, 750, 825, 887])
    tax_cents = (subtotal_cents * tax_bps + 5000) // 10000
    total_cents = subtotal_cents + tax_cents - credits_cents

    gt = GroundTruthInvoice(
        invoice_id=f"{index:03d}_{layout}",
        layout=layout,
        vendor_name=vendor,
        vendor_category=category,
        invoice_no=invoice_no,
        invoice_date=invoice_date,
        currency="USD",
        line_items=tuple(all_items),
        subtotal_cents=subtotal_cents,
        tax_cents=tax_cents,
        credits_cents=credits_cents,
        total_cents=total_cents,
        pdf_path=pdf_rel_path,
        group_subtotals_cents=tuple(group_subtotals),
    )
    return gt, sections


def _draw_header_block(
    c: Canvas,
    gt: GroundTruthInvoice,
    address: str,
    tax_id: str,
    *,
    jitter_rng: random.Random | None = None,
) -> float:
    """Draw invoice vendor & metadata header; return top Y coordinate for table header."""

    def jx(val: float) -> float:
        return val + (jitter_rng.uniform(-1.5, 1.5) if jitter_rng else 0.0)

    def jy(val: float) -> float:
        return val + (jitter_rng.uniform(-1.5, 1.5) if jitter_rng else 0.0)

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 15)
    c.drawString(jx(MARGIN_X), jy(742), gt.vendor_name)

    c.setFillColor(colors.HexColor("#4B5563"))
    c.setFont("Helvetica", 9)
    c.drawString(jx(MARGIN_X), jy(726), address)
    c.drawString(jx(MARGIN_X), jy(713), f"Tax ID: {tax_id}  |  Terms: Net 30")

    c.setFillColor(colors.HexColor("#1D4ED8"))
    c.setFont("Helvetica-Bold", 18)
    c.drawRightString(jx(PAGE_WIDTH - MARGIN_X), jy(742), "INVOICE")

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 9.5)
    c.drawRightString(jx(470), jy(720), "Invoice No:")
    c.drawRightString(jx(470), jy(706), "Invoice Date:")
    c.drawRightString(jx(470), jy(692), "PO Reference:")

    c.setFont("Helvetica", 9.5)
    c.drawRightString(jx(PAGE_WIDTH - MARGIN_X), jy(720), gt.invoice_no)
    c.drawRightString(jx(PAGE_WIDTH - MARGIN_X), jy(706), gt.invoice_date)
    po_num = f"PO-{int(gt.invoice_no.split('-')[-1]) + 4000}"
    c.drawRightString(jx(PAGE_WIDTH - MARGIN_X), jy(692), po_num)

    # Bill To box
    c.setStrokeColor(colors.HexColor("#E5E7EB"))
    c.setLineWidth(0.75)
    c.line(MARGIN_X, 680, PAGE_WIDTH - MARGIN_X, 680)
    c.setFillColor(colors.HexColor("#374151"))
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(jx(MARGIN_X), jy(666), "BILL TO:")
    c.setFont("Helvetica", 8.5)
    c.drawString(jx(MARGIN_X + 48), jy(666), "Northstar Operations Purchasing Dept.")
    c.drawString(jx(MARGIN_X + 48), jy(654), "900 Harborview Way, Suite 120, Baltimore, MD 21202")
    return 632.0


def _draw_table_header(
    c: Canvas,
    y: float,
    col_x: dict[str, float],
    *,
    wrapped: bool = False,
    jitter_rng: random.Random | None = None,
) -> float:
    band_h = 28.0 if wrapped else 20.0
    c.setFillColor(colors.HexColor("#F3F4F6"))
    c.rect(MARGIN_X, y - band_h, PAGE_WIDTH - 2 * MARGIN_X, band_h, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#9CA3AF"))
    c.setLineWidth(0.8)
    c.line(MARGIN_X, y, PAGE_WIDTH - MARGIN_X, y)
    c.line(MARGIN_X, y - band_h, PAGE_WIDTH - MARGIN_X, y - band_h)

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 8.5)

    def jx(val: float) -> float:
        return val + (jitter_rng.uniform(-1.2, 1.2) if jitter_rng else 0.0)

    def jy(val: float) -> float:
        return val + (jitter_rng.uniform(-1.2, 1.2) if jitter_rng else 0.0)

    if wrapped:
        c.drawString(col_x["sku"], y - 11, "ITEM")
        c.drawString(col_x["sku"], y - 22, "CODE")
        c.drawString(col_x["desc"], y - 11, "PRODUCT")
        c.drawString(col_x["desc"], y - 22, "DESCRIPTION")
        c.drawRightString(col_x["qty"], y - 11, "QTY /")
        c.drawRightString(col_x["qty"], y - 22, "UNIT")
        c.drawRightString(col_x["unit_price"], y - 11, "UNIT")
        c.drawRightString(col_x["unit_price"], y - 22, "RATE")
        c.drawRightString(col_x["amount"], y - 11, "EXT'D")
        c.drawRightString(col_x["amount"], y - 22, "NET")
    else:
        c.drawString(jx(col_x["sku"]), jy(y - 13), "SKU")
        c.drawString(jx(col_x["desc"]), jy(y - 13), "Description")
        c.drawRightString(jx(col_x["qty"]), jy(y - 13), "Qty")
        c.drawRightString(jx(col_x["unit_price"]), jy(y - 13), "Unit Price")
        c.drawRightString(jx(col_x["amount"]), jy(y - 13), "Amount")

    return y - band_h - 6.0


def _draw_jittered_words_left(c: Canvas, x: float, y: float, text: str, rng: random.Random) -> None:
    cursor = x
    for word in text.split():
        dx = rng.uniform(-2.5, 2.5)
        dy = rng.uniform(-2.5, 2.5)
        c.drawString(cursor + dx, y + dy, word)
        cursor += c.stringWidth(word, "Helvetica", 8.8) + 6.2


def _draw_summary_block(
    c: Canvas,
    y: float,
    gt: GroundTruthInvoice,
    *,
    parens_negative: bool = False,
    jitter_rng: random.Random | None = None,
) -> None:
    def jx(val: float) -> float:
        return val + (jitter_rng.uniform(-1.8, 1.8) if jitter_rng else 0.0)

    def jy(val: float) -> float:
        return val + (jitter_rng.uniform(-1.8, 1.8) if jitter_rng else 0.0)

    label_x = 468.0
    val_x = PAGE_WIDTH - MARGIN_X
    c.setStrokeColor(colors.HexColor("#9CA3AF"))
    c.setLineWidth(0.75)
    c.line(350.0, y + 6, val_x, y + 6)

    cur_y = y - 10
    c.setFont("Helvetica", 9.5)
    c.setFillColor(colors.HexColor("#111827"))
    c.drawRightString(jx(label_x), jy(cur_y), "Subtotal:")
    c.drawRightString(jx(val_x), jy(cur_y), format_cents(gt.subtotal_cents))

    if gt.credits_cents > 0:
        cur_y -= 16
        c.drawRightString(jx(label_x), jy(cur_y), "Credits Applied:")
        c.drawRightString(
            jx(val_x),
            jy(cur_y),
            format_cents(-gt.credits_cents, parens_negative=parens_negative),
        )

    cur_y -= 16
    c.drawRightString(jx(label_x), jy(cur_y), "Tax:")
    c.drawRightString(jx(val_x), jy(cur_y), format_cents(gt.tax_cents))

    cur_y -= 20
    c.setStrokeColor(colors.HexColor("#111827"))
    c.setLineWidth(1.0)
    c.line(350.0, cur_y + 13, val_x, cur_y + 13)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawRightString(jx(label_x), jy(cur_y), "Grand Total:")
    c.drawRightString(jx(val_x), jy(cur_y), format_cents(gt.total_cents))


def _render_invoice_pdf(
    pdf_path: Path,
    gt: GroundTruthInvoice,
    sections: list[tuple[str, list[GroundTruthLineItem]]],
    rng: random.Random,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = _VENDOR_CATALOG[gt.layout]
    address = str(catalog["address"])
    tax_id = str(catalog["tax_id"])

    col_x = {
        "sku": MARGIN_X + 4.0,
        "desc": MARGIN_X + 76.0,
        "qty": 388.0,
        "unit_price": 468.0,
        "amount": PAGE_WIDTH - MARGIN_X - 4.0,
    }

    is_wrapped = gt.layout == "wrapped_header"
    is_jittered = gt.layout == "jittered_ocr_layout"
    is_multipage = gt.layout == "multipage_table"
    jitter_rng = rng if is_jittered else None

    c = Canvas(str(pdf_path), pagesize=LETTER)
    c.setTitle(f"Invoice {gt.invoice_no}")
    y = _draw_header_block(c, gt, address, tax_id, jitter_rng=jitter_rng)
    y = _draw_table_header(c, y, col_x, wrapped=is_wrapped, jitter_rng=jitter_rng)

    page_num = 1
    total_pages = 2 if is_multipage else 1

    def draw_footer(p_num: int, continuation: bool = False) -> None:
        c.setFillColor(colors.HexColor("#6B7280"))
        c.setFont("Helvetica", 7.8)
        if is_jittered:
            c.drawString(
                MARGIN_X,
                34,
                f"OCR-CONF: 0.96  SCAN-BATCH: B{gt.invoice_id[:3]}  REF: {gt.invoice_no}",
            )
            c.drawRightString(
                PAGE_WIDTH - MARGIN_X,
                34,
                "REFERENCE SYNTHETIC COPY - VERIFY AGAINST REMITTANCE LEDGER",
            )
        else:
            if continuation:
                c.drawString(MARGIN_X, 52, "Continued on Page 2...")
            c.drawRightString(
                PAGE_WIDTH - MARGIN_X,
                34,
                f"Page {p_num} of {total_pages}  |  Reference synthetic invoice",
            )

    bottom_floor = 240.0 if is_multipage else 128.0
    max_desc_chars = 36 if is_wrapped else 43
    parens_style = gt.layout == "credits_discounts" and (int(gt.invoice_id[:3]) % 2 == 1)

    for sec_idx, (sec_title, items) in enumerate(sections):
        if sec_title:
            c.setFillColor(colors.HexColor("#1E3A8A"))
            c.setFont("Helvetica-Bold", 9.0)
            c.drawString(col_x["sku"], y - 8, sec_title)
            y -= 18.0

        for row_idx, item in enumerate(items):
            desc_lines = _wrap_text(item.description, max_desc_chars)
            row_h = max(20.0, 12.0 * len(desc_lines) + 8.0)
            if is_multipage:
                row_h = max(27.0, 12.0 * len(desc_lines) + 12.0)
            elif is_jittered:
                row_h = max(25.0, 12.0 * len(desc_lines) + 12.0)

            if is_multipage and y - row_h < bottom_floor:
                draw_footer(page_num, continuation=True)
                c.showPage()
                page_num += 1
                c.setFillColor(colors.HexColor("#111827"))
                c.setFont("Helvetica-Bold", 11)
                c.drawString(MARGIN_X, 748, f"{gt.vendor_name} — {gt.invoice_no} (Continued)")
                y = _draw_table_header(c, 732.0, col_x, wrapped=False)

            if gt.layout == "standard_grid":
                c.setStrokeColor(colors.HexColor("#E5E7EB"))
                c.setLineWidth(0.5)
                c.line(MARGIN_X, y - row_h + 2, PAGE_WIDTH - MARGIN_X, y - row_h + 2)
                for vx in (col_x["desc"] - 6, col_x["qty"] - 42, col_x["unit_price"] - 72):
                    c.line(vx, y + 6, vx, y - row_h + 2)

            baseline_y = y - 10.0
            row_drift = rng.uniform(-2.5, 2.5) if is_jittered else 0.0
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont("Helvetica", 8.8)

            use_parens = parens_style ^ (row_idx % 2 == 1) if item.is_credit else parens_style
            qty_str = (
                str(int(item.qty))
                if item.qty == item.qty.to_integral_value()
                else f"{item.qty:.1f}"
            )
            unit_str = format_cents(item.unit_price_cents, parens_negative=use_parens)
            amt_str = format_cents(item.amount_cents, parens_negative=use_parens)

            if is_jittered:
                _draw_jittered_words_left(c, col_x["sku"], baseline_y + row_drift, item.sku, rng)
                for l_idx, line in enumerate(desc_lines):
                    _draw_jittered_words_left(
                        c,
                        col_x["desc"],
                        baseline_y + row_drift - l_idx * 11.5,
                        line,
                        rng,
                    )
                c.drawRightString(
                    col_x["qty"] + rng.uniform(-2.5, 2.5),
                    baseline_y + row_drift + rng.uniform(-2.5, 2.5),
                    qty_str,
                )
                c.drawRightString(
                    col_x["unit_price"] + rng.uniform(-2.5, 2.5),
                    baseline_y + row_drift + rng.uniform(-2.5, 2.5),
                    unit_str,
                )
                c.drawRightString(
                    col_x["amount"] + rng.uniform(-2.5, 2.5),
                    baseline_y + row_drift + rng.uniform(-2.5, 2.5),
                    amt_str,
                )
            else:
                c.drawString(col_x["sku"], baseline_y, item.sku)
                for l_idx, line in enumerate(desc_lines):
                    c.drawString(col_x["desc"], baseline_y - l_idx * 11.5, line)
                c.drawRightString(col_x["qty"], baseline_y, qty_str)
                c.drawRightString(col_x["unit_price"], baseline_y, unit_str)
                c.drawRightString(col_x["amount"], baseline_y, amt_str)

            y -= row_h

        if sec_title:
            sec_sum = sum(it.amount_cents for it in items)
            c.setStrokeColor(colors.HexColor("#CBD5E1"))
            c.setLineWidth(0.6)
            c.line(col_x["desc"], y + 2, PAGE_WIDTH - MARGIN_X, y + 2)
            c.setFillColor(colors.HexColor("#1E293B"))
            c.setFont("Helvetica-Bold", 8.8)
            label_variants = [
                f"{sec_title.split(':')[0]} Subtotal",
                f"{sec_title.split(':')[0]} Summary Total",
                f"Group {chr(65 + sec_idx)} Package Total",
            ]
            sec_label = label_variants[sec_idx % len(label_variants)]
            c.drawString(col_x["desc"], y - 10, sec_label)
            c.drawRightString(col_x["amount"], y - 10, format_cents(sec_sum))
            y -= 24.0

    if y < 120.0:
        draw_footer(page_num, continuation=True)
        c.showPage()
        page_num += 1
        y = 720.0

    _draw_summary_block(
        c,
        y - 8.0,
        gt,
        parens_negative=parens_style,
        jitter_rng=jitter_rng,
    )
    draw_footer(page_num, continuation=False)
    c.save()


def generate_dataset(
    output_dir: Path,
    *,
    count: int = 120,
    seed: int = 42,
) -> list[GroundTruthInvoice]:
    """Generate `count` deterministic synthetic PDFs + JSON ground truth files."""
    if count <= 0:
        raise ValueError("count must be positive.")

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir = output_dir / "pdfs"
    gt_dir = output_dir / "ground_truth"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[GroundTruthInvoice] = []
    for idx in range(1, count + 1):
        layout = LAYOUT_TYPES[(idx - 1) % len(LAYOUT_TYPES)]
        invoice_id = f"{idx:03d}_{layout}"
        pdf_rel = f"pdfs/{invoice_id}.pdf"
        pdf_path = output_dir / pdf_rel
        gt_path = gt_dir / f"{invoice_id}.json"

        gt, sections = _build_ground_truth(idx, layout, rng, pdf_rel)
        _render_invoice_pdf(pdf_path, gt, sections, rng)

        gt_path.write_text(json.dumps(gt.to_dict(), indent=2) + "\n", encoding="utf-8")
        manifest.append(gt)

    manifest_path = output_dir / "manifest.json"
    manifest_payload = {
        "seed": seed,
        "count": count,
        "layouts": list(LAYOUT_TYPES),
        "invoices": [gt.to_dict() for gt in manifest],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return manifest
