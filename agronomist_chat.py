"""
AI Agronomist Chatbot Assistant ('Krishi Mitra / Crop Doctor')
Provides practical guidance on cotton crop diseases, chemical tank-mixes,
spray timing, organic recipes, and soil fertility management.
Operates seamlessly offline with knowledge base heuristics, and optionally
leverages Groq or Gemini API when keys are provided.
"""

import os
import json
import re

# Curated Agronomic Knowledge Base with pattern matching
AGRONOMY_KB = [
    {
        "patterns": [r"tank\s*mix", r"mix.*tank", r"mix.*chemical", r"combine.*spray", r"cocktail", r"mix.*together", r"can\s*i\s*mix"],
        "answer": (
            "**Tank Mixing Guidelines for Cotton:**\n\n"
            "• **Safe Combinations:** Copper Oxychloride (50% WP) + Streptocycline (100 ppm) is the gold standard mix for Bacterial Blight.\n"
            "• **Fungicide + Insecticide:** Most WP/EC fungicides (e.g., Mancozeb) can be mixed with synthetic pyrethroids or organophosphates, but NEVER mix with alkaline solutions like lime sulphur or Bordeaux mixture.\n"
            "• **Foliar Nutrients:** Magnesium Sulphate (1%) and Urea (1%) can be mixed together for rapid recovery from Leaf Redding.\n"
            "• **Jar Test:** Always perform a quick jar test: mix small quantities in a clear jar and wait 15 minutes. If curdling, sedimentation, or foaming occurs, do NOT spray."
        )
    },
    {
        "patterns": [r"rain", r"wash\s*off", r"rainfast", r"cloudy", r"weather.*spray", r"spray.*weather"],
        "answer": (
            "**Rainfastness & Weather Precautions:**\n\n"
            "• **Contact Fungicides (Mancozeb, Copper):** Require at least **3 to 4 hours** of dry leaf surface to form a protective film.\n"
            "• **Systemic Fungicides (Propiconazole, Carbendazim):** Absorbed by leaf tissue within **1 to 2 hours**.\n"
            "• **Sticker / Spreader:** Always add a non-ionic adjuvant or wetting agent (e.g., Sandovit, Agrowet @ 0.5 ml/L) during rainy seasons to prevent wash-off.\n"
            "• Avoid spraying during peak afternoon heat (>35°C) to prevent chemical leaf scorching."
        )
    },
    {
        "patterns": [r"whitefly", r"curl\s*virus", r"vector", r"safed\s*makk?hi", r"chidiya"],
        "answer": (
            "**Whitefly & Leaf Curl Virus Control:**\n\n"
            "• **Vector Impact:** Whitefly (*Bemisia tabaci*) transmits Cotton Leaf Curl Virus (CLCuV). Once infected, the virus cannot be cured, so vector control is paramount.\n"
            "• **Yellow Sticky Traps:** Install 15–20 yellow sticky traps per acre at crop canopy height.\n"
            "• **Insecticide Spray:** Use Diafenthiuron 50% WP (1.2 g/L) or Afidopyropen 50 g/L (2 ml/L) targeting the underside of leaves.\n"
            "• **Organic Spray:** Spray 5% Neem Seed Kernel Extract (NSKE) early in the season to disrupt nymph feeding."
        )
    },
    {
        "patterns": [r"lal\s*patti", r"red\s*leaf", r"redding", r"magnesium", r"deficiency", r"nutrient"],
        "answer": (
            "**Managing Leaf Redding (Lal Patti) in Cotton:**\n\n"
            "• **Root Cause:** Nutritional imbalance (Magnesium and Nitrogen deficiency) coupled with high boll load and sudden cool night temperatures.\n"
            "• **Foliar Treatment:** Spray **Magnesium Sulphate (MgSO4 @ 10 g/L)** + **Potassium Nitrate (13:0:45 @ 10 g/L)** or **Urea (10 g/L)**.\n"
            "• **Schedule:** Apply two sprays at a 15-day interval during flowering and early boll development.\n"
            "• **Drainage:** Ensure field drainage is clear, as waterlogged roots cannot absorb magnesium."
        )
    },
    {
        "patterns": [r"bacterial", r"blight", r"angular", r"blackarm", r"streptocycline"],
        "answer": (
            "**Bacterial Blight (Blackarm) Protocol:**\n\n"
            "• **Pathogen:** *Xanthomonas citri pv. malvacearum* spread by windblown rain and tool contact.\n"
            "• **Treatment:** Spray Copper Oxychloride 50% WP (2.5 g/L) combined with Streptocycline (1 g per 10 L of water).\n"
            "• **Field Hygiene:** Never enter or cultivate in cotton fields when the crop canopy is wet from rain or dew."
        )
    },
    {
        "patterns": [r"organic", r"neem", r"trichoderma", r"panchagavya", r"bio", r"desi", r"jeevamrut"],
        "answer": (
            "**Organic & Biological Management:**\n\n"
            "• **Soil Treatment:** Mix 2.5 kg *Trichoderma viride* in 500 kg farmyard manure (FYM), incubate under shade for 7 days, and broadcast per acre.\n"
            "• **Foliar Bio-Spray:** 5% Neem Seed Kernel Extract (NSKE) or *Pseudomonas fluorescens* (5 g/L) suppresses both bacterial and early fungal spore development.\n"
            "• **Immunity Booster:** Fermented sour buttermilk (5 L) + cow urine (5 L) in 100 L water sprayed during vegetative stages strengthens leaf epidermis."
        )
    },
    {
        "patterns": [r"water", r"irrigation", r"paani", r"sinchai", r"per\s*day", r"drip"],
        "answer": (
            "**Cotton Crop Water & Irrigation Management:**\n\n"
            "• **Total Water Requirement:** A cotton crop typically requires **500 to 700 mm** of water across its life cycle (approx. **3 to 5 mm per day** during peak growth).\n"
            "• **Critical Growth Stages for Watering:**\n"
            "  1. *Square Formation (35-45 DAS):* Maintain moderate soil moisture; avoid excessive moisture which delays flowering.\n"
            "  2. *Flowering & Peak Boll Development (55-90 DAS):* **Most critical period**; drought stress causes heavy flower square and young boll shedding.\n"
            "• **Drip Irrigation Rate:** In drip systems, supply **2 to 3 liters per plant per day** during vegetative phase, increasing to **4 to 5 liters per plant per day** during peak flowering/boll load.\n"
            "• **Avoid Waterlogging:** Cotton roots are highly sensitive to standing water. Always ensure proper drainage to prevent root asphyxiation and sudden wilt."
        )
    },
    {
        "patterns": [r"dosage", r"spray\s*rate", r"pump", r"kitna\s*daale", r"knapsack", r"per\s*15l", r"dawa.*dose"],
        "answer": (
            "**Standard 15-Liter Knapsack Sprayer Dosages:**\n\n"
            "• **Mancozeb 75% WP:** 35–40 g per 15-L pump (2.5 g/L).\n"
            "• **Copper Oxychloride:** 35–40 g per 15-L pump.\n"
            "• **Streptocycline:** 1.5 g per 15-L pump.\n"
            "• **Propiconazole 25% EC:** 15 ml per 15-L pump (1 ml/L).\n"
            "• **Magnesium Sulphate:** 150 g per 15-L pump (10 g/L).\n"
            "• **Neem Oil (10,000 ppm):** 30–45 ml per 15-L pump with detergent emulsifier."
        )
    }
]


def ask_agronomist(query, current_disease=None, disease_info=None, api_key=None):
    """
    Answers farmer queries using contextual diagnosis data and agricultural knowledge base.
    """
    clean_q = query.lower().strip()

    # Answer specific questions before falling back to the current diagnosis.
    for entry in AGRONOMY_KB:
        if any(re.search(pattern, clean_q) for pattern in entry["patterns"]):
            return entry["answer"]

    # If current disease is available and user asks about treatment/control
    if current_disease and disease_info and any(w in clean_q for w in ["treatment", "cure", "medicine", "dawa", "control", "spray", "prevent"]):
        info = disease_info.get(current_disease, {})
        chems = "\n".join([f"• {c}" for c in info.get("chemical_control", [])])
        acts = "\n".join([f"• {a}" for a in info.get("immediate_actions", [])])
        orgs = "\n".join([f"• {o}" for o in info.get("organic_biological", [])])
        return (
            f"### 🩺 Prescription for {current_disease}\n\n"
            f"**Recommended Chemical Sprays:**\n{chems}\n\n"
            f"**Immediate Actions:**\n{acts}\n\n"
            f"**Biological & Organic Options:**\n{orgs}"
        )

    # Fallback to general advisory
    if current_disease and disease_info:
        info = disease_info.get(current_disease, {})
        return (
            f"Based on your current diagnosis (**{current_disease}**):\n"
            f"• **Severity:** {info.get('severity', 'Moderate')}\n"
            f"• **Pathogen:** {info.get('pathogen', 'N/A')}\n\n"
            f"**Quick Advice:** Inspect surrounding plants, isolate affected rows, and apply "
            f"the recommended chemical or organic spray listed in the diagnostic tabs above."
        )

    return (
        "**Crop Doctor Advice:**\n\n"
        "For optimal cotton crop health, ensure proper plant spacing (90x60 cm), balanced NPK fertilization, "
        "and avoid waterlogging. If you suspect disease, upload a photo above to receive a full clinical diagnosis, "
        "or ask specific questions regarding spray dosages, tank mixing, or whitefly control."
    )
