"""
Professional Agronomic PDF Prescription Report Generator
Generates clinical A4 diagnosis sheets with leaf thumbnails, chemical spray schedules,
and organic management protocols.
"""

import io
import os
from datetime import datetime
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)


def generate_pdf_report(analysis_result, original_image=None, specimen_label="Field Sample"):
    """
    Generates a professional PDF report bytes buffer from analysis results.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1b5e20'),
        alignment=0,
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#555555'),
        spaceAfter=10
    )
    section_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1b5e20'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#222222')
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11.5,
        leftIndent=12,
        firstLineIndent=-8,
        textColor=colors.HexColor('#222222'),
        spaceAfter=2
    )
    alert_style = ParagraphStyle(
        'DocAlert',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    # ---------------------------------------------------------
    # 1. HEADER
    # ---------------------------------------------------------
    story.append(Paragraph("🌿 COTTON PATHOLOGY CLINICAL REPORT", title_style))
    story.append(Paragraph("AI-Powered Agricultural Decision Support & Agronomic Prescription Advisory", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2e7d32'), spaceAfter=8))

    # ---------------------------------------------------------
    # 2. SPECIMEN METADATA & IMAGES TABLE
    # ---------------------------------------------------------
    info = analysis_result.get("disease_info", {}) or {}
    pred_class = analysis_result.get("predicted_class", "Unknown")
    confidence = analysis_result.get("confidence", 0.0)
    is_cotton = analysis_result.get("is_cotton", True)
    sev_level = info.get("severity", "Moderate")

    meta_data = [
        [
            Paragraph("<b>Date & Time:</b>", body_style),
            Paragraph(datetime.now().strftime('%d %B %Y, %I:%M %p'), body_style),
            Paragraph("<b>Cotton Verified:</b>", body_style),
            Paragraph("Yes (Pass)" if is_cotton else "No (Alert)", body_style),
        ],
        [
            Paragraph("<b>Specimen:</b>", body_style),
            Paragraph(specimen_label, body_style),
            Paragraph("<b>Risk Severity:</b>", body_style),
            Paragraph(f"<b>{sev_level.upper()}</b>", body_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[80, 190, 95, 175])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f8e9')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#c5e1a5')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e8f5e9')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Optional Images side-by-side
    img_cells = []
    if original_image is not None:
        try:
            temp_orig_path = io.BytesIO()
            orig_thumb = original_image.copy().convert("RGB")
            orig_thumb.thumbnail((160, 160))
            orig_thumb.save(temp_orig_path, format="JPEG", quality=85)
            temp_orig_path.seek(0)
            rl_orig = RLImage(temp_orig_path, width=110, height=110)
            img_cells.append([rl_orig, Paragraph("<b>Uploaded Specimen</b>", body_style)])
        except Exception:
            pass

    heatmap = analysis_result.get("heatmap")
    if heatmap is not None:
        try:
            temp_heat_path = io.BytesIO()
            heat_thumb = heatmap.copy().convert("RGB")
            heat_thumb.thumbnail((160, 160))
            heat_thumb.save(temp_heat_path, format="JPEG", quality=85)
            temp_heat_path.seek(0)
            rl_heat = RLImage(temp_heat_path, width=110, height=110)
            img_cells.append([rl_heat, Paragraph("<b>Lesion Saliency Heatmap</b>", body_style)])
        except Exception:
            pass

    if img_cells:
        img_table_data = [
            [cell[0] for cell in img_cells],
            [cell[1] for cell in img_cells]
        ]
        img_table = Table(img_table_data, colWidths=[140] * len(img_cells))
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 3. DIAGNOSTIC FINDING BANNER
    # ---------------------------------------------------------
    sev_color_map = {
        "Normal": colors.HexColor('#2e7d32'),
        "Low": colors.HexColor('#7cb342'),
        "Moderate": colors.HexColor('#f57c00'),
        "High": colors.HexColor('#d32f2f'),
        "Critical": colors.HexColor('#b71c1c')
    }
    banner_color = sev_color_map.get(sev_level, colors.HexColor('#2e7d32'))

    banner_data = [[
        Paragraph(f"DIAGNOSIS: {pred_class.upper()} &nbsp;|&nbsp; CONFIDENCE: {confidence:.2f}% &nbsp;|&nbsp; RISK: {sev_level.upper()}", alert_style)
    ]]
    banner_table = Table(banner_data, colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('CORNERPAD', (0, 0), (-1, -1), 4),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 6))

    # Pathogen & Category
    pathogen = info.get("pathogen", "N/A")
    cat = info.get("category", "General")
    story.append(Paragraph(f"<b>Category:</b> {cat} &nbsp;&bull;&nbsp; <b>Causal Agent / Pathogen:</b> <i>{pathogen}</i>", body_style))
    if "overview" in info:
        story.append(Paragraph(f"<b>Overview:</b> {info['overview']}", body_style))
    story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 4. CHEMICAL PRESCRIPTION TABLE
    # ---------------------------------------------------------
    chem_list = info.get("chemical_control", [])
    if chem_list:
        story.append(Paragraph("💊 RECOMMENDED CHEMICAL CONTROL & DOSAGE SCHEDULE", section_style))
        chem_rows = [[
            Paragraph("<b>#</b>", body_style),
            Paragraph("<b>Active Ingredient / Formulation & Recommended Water Dilution Rate</b>", body_style)
        ]]
        for idx, item in enumerate(chem_list, 1):
            chem_rows.append([
                Paragraph(str(idx), body_style),
                Paragraph(item, body_style)
            ])

        chem_table = Table(chem_rows, colWidths=[24, 516])
        chem_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0f2f1')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#80cbc4')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(chem_table)
        story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 5. IMMEDIATE ACTION & ORGANIC MANAGEMENT
    # ---------------------------------------------------------
    act_list = info.get("immediate_actions", [])
    org_list = info.get("organic_biological", [])

    if act_list or org_list:
        dual_col_data = [
            [
                Paragraph("⚡ <b>Immediate Containment (24-48h)</b>", section_style),
                Paragraph("🌱 <b>Organic & Biological Solutions</b>", section_style)
            ]
        ]
        
        act_text = "<br/>".join([f"&bull; {a}" for a in act_list[:4]])
        org_text = "<br/>".join([f"&bull; {o}" for o in org_list[:4]])
        
        dual_col_data.append([
            Paragraph(act_text, body_style),
            Paragraph(org_text, body_style)
        ])

        dual_table = Table(dual_col_data, colWidths=[265, 275])
        dual_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(dual_table)
        story.append(Spacer(1, 6))

    # ---------------------------------------------------------
    # 6. DIFFERENTIAL CANDIDATES & WEATHER
    # ---------------------------------------------------------
    top3 = analysis_result.get("top_3", [])
    if len(top3) > 1:
        diff_str = " | ".join([f"{c['class_name']}: {c['probability']:.1f}%" for c in top3])
        story.append(Paragraph(f"<b>Differential Diagnosis (Top Probability Spread):</b> {diff_str}", body_style))

    if "favorable_weather" in info:
        story.append(Paragraph(f"<b>Favorable Weather Triggers:</b> {info['favorable_weather']}", body_style))

    # ---------------------------------------------------------
    # 7. DISCLAIMER & SIGN-OFF
    # ---------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=4))
    story.append(Paragraph(
        "<b>ADVISORY DISCLAIMER:</b> This diagnostic report is generated by an artificial intelligence decision-support tool. "
        "Field factors such as root status, pest complexes, and soil tests should be considered before massive commercial chemical applications. "
        "Consult your local Krishi Vigyan Kendra (KVK) or State Agricultural University extension officers.",
        subtitle_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
