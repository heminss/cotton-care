# 🌿 Cotton Leaf Disease AI Prediction & Agronomic Diagnosis System

An end-to-end artificial intelligence decision-support platform designed for cotton farmers, agronomists, and agricultural researchers. It integrates a **two-stage deep learning pipeline** with an **authoritative clinical agronomy knowledge base** to detect, verify, and diagnose 9 distinct cotton leaf conditions with immediate, practical treatment schedules.

---

## 🚀 Key Features & Capabilities

### 1. 🛡️ Two-Stage AI Verification Pipeline
- **Stage 1: Gatekeeper Model (`cotton_detector.tflite`)**: Evaluates input images to verify whether the object is a genuine cotton leaf before performing diagnosis. Rejects non-plant objects, non-cotton foliage, and arbitrary uploads to eliminate out-of-distribution false positives.
- **Stage 2: Pathology Specialist (`cotton_leaf_disease.tflite`)**: 9-class deep convolutional classifier trained on extensive agricultural imagery to deliver class probabilities with sub-100ms inference.

### 2. 🩺 Comprehensive Agronomic Diagnosis & Prescriptions
Every condition includes complete treatment protocols:
- **Pathogen / Etiology**: Scientific causal agent (fungal, bacterial, viral, nutrient deficiency, chemical drift).
- **Visual Symptoms**: Characteristic indicators, leaf lesion shapes, chlorotic halos, and vascular markers.
- **Severity Ranking**: Color-coded risk indicators (Normal, Moderate, High, Critical).
- **Immediate Containment**: Emergency 24-48 hour containment steps.
- **Chemical Control & Dosage**: Specific active ingredients (e.g. *Mancozeb*, *Propiconazole*, *Streptocycline*, *Copper Oxychloride*, *Diafenthiuron*) with exact dilution rates per liter of water.
- **Organic & Biological Solutions**: *Trichoderma viride*, *Pseudomonas fluorescens*, 5% Neem Seed Kernel Extract (NSKE), bio-stimulants.
- **Cultural Prevention**: Crop rotation, resistant hybrids, seed treatment, and spacing guidelines.

### 3. 📄 Clinical PDF Prescription Report Generator
- 1-Click generation of a formatted **A4 Clinical Prescription Report (PDF)** complete with specimen photo, lesion heatmap, diagnosis badge, confidence meter, exact chemical spray table with dosage per liter, organic recipe, and doctor/agronomist signature block.
- Ready for printing or instant sharing via WhatsApp with agricultural extension officers.

### 4. 💬 AI Crop Doctor Chatbot Assistant ("Krishi Mitra")
- Interactive farming assistant providing guidance on:
  - Chemical compatibility & tank-mixing safety (e.g. *Can I mix Mancozeb & Streptocycline?*).
  - Spray timings & rainfastness intervals.
  - Sucking pest & Whitefly vector management.
  - Knapsack sprayer dosage calculations (per 15-liter pump).
  - Leaf Redding (Lal Patti) management.

### 5. 🌾 Multi-Leaf Field Survey Mode (Batch Scanning)
- Allows uploading multiple leaf photos from different quadrants of a cotton field.
- Automatically computes:
  - Whole-field infection incidence percentage.
  - Dominant pathology across the plot.
  - Targeted whole-plot vs spot-treatment containment recommendations.

### 6. 🌐 Multi-Lingual Support
Designed for frontline Indian and global cotton growers with native UI and clinical summaries in:
- **English**
- **हिंदी (Hindi)**
- **मराठी (Marathi)**
- **ગુજરાતી (Gujarati)**

### 7. 🔍 Explainable AI & Visual Saliency
- Highlights areas of severe lesion concentration, chlorosis, and necrosis directly on the uploaded image using a visual focus heatmap overlay.

---

## 🦠 Supported Conditions (9 Classes)

| Condition | Category | Causal Agent / Pathogen | Severity |
| :--- | :--- | :--- | :--- |
| **Alternaria Leaf Spot** | Fungal | *Alternaria macrospora / A. alternata* | Moderate |
| **Bacterial Blight** | Bacterial | *Xanthomonas citri pv. malvacearum* | High |
| **Curl Virus** | Viral | *Cotton Leaf Curl Virus* (Vector: Whitefly) | Critical |
| **Fusarium Wilt** | Soilborne Fungi | *Fusarium oxysporum f. sp. vasinfectum* | High |
| **Healthy Leaf** | Healthy | Pathogen-free foliage | Normal |
| **Herbicide Growth Damage** | Abiotic / Chemical | Herbicide drift (2,4-D, glyphosate) | Moderate |
| **Leaf Redding** | Physiological | Mg/N deficiency, anthocyanin accumulation | Moderate |
| **Leaf Variegation** | Genetic / Minor Virus | Genetic chimera, calico virus, micronutrient | Low |
| **Verticillium Wilt** | Soilborne Fungi | *Verticillium dahliae* | High |

---

## 📂 Project Structure

```
newmodell/
├── app.py                      # Modern Streamlit Web Application
├── inference_engine.py         # Dual-stage AI engine & image preprocessor
├── pdf_report_generator.py     # Clinical A4 PDF prescription generator
├── agronomist_chat.py          # AI Crop Doctor Q&A knowledge base
├── disease_info.json           # Comprehensive agronomic database with 4 languages
├── class_names_9class.json     # Class index mapping (9 classes)
├── class_names.json            # Backward compatibility alias
├── cotton_detector.tflite      # Gatekeeper model (Cotton vs Non-Cotton)
├── cotton_leaf_disease.tflite  # 9-class deep pathology classification model
├── best_cotton_detector_v2.keras # Keras archive for cotton detector
├── cotton_leaf_model_9class.keras # Keras archive for 9-class model
├── test_engine.py              # Automated test suite (7 unit tests)
├── requirements.txt            # Python dependencies
├── run_app.bat                 # One-click Windows desktop launcher
└── sample_images/              # Built-in test samples
    ├── healthy_leaf.jpg
    ├── bacterial_blight.jpg
    └── non_cotton_ficus.jpg
```

---

## ⚡ Quick Start Guide

### Option 1: 1-Click Windows Launch
Double-click [`run_app.bat`](file:///C:/Users/Techn/OneDrive/Desktop/newmodell/run_app.bat) to launch the web dashboard in your browser.

### Option 2: Command Line
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run automated verification:
   ```bash
   python test_engine.py
   ```
3. Start the application:
   ```bash
   streamlit run app.py
   ```
