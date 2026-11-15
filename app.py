"""
🌿 Cotton Leaf Disease Prediction & Clinical Agronomic Diagnosis
Two-Stage AI Verification Pipeline Powered by Google LiteRT
Includes PDF Prescription Export, AI Agronomist Q&A, and Multi-Leaf Field Survey
"""

import os
import json
from datetime import datetime
from PIL import Image, ImageOps
import streamlit as st

from inference_engine import CottonInferenceEngine
from pdf_report_generator import generate_pdf_report
from agronomist_chat import ask_agronomist

# ---------------------------------------------------------
# PAGE SETUP & MODERN AGRICULTURAL STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cotton Leaf Disease AI Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# INITIALIZE INFERENCE ENGINE
# ---------------------------------------------------------
@st.cache_resource
def get_engine():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return CottonInferenceEngine(base_dir=current_dir)

try:
    engine = get_engine()
except (OSError, RuntimeError, ValueError) as exc:
    st.error(f"Unable to load the analysis models: {exc}")
    st.stop()

# Multi-lingual UI strings
UI_TEXTS = {
    "en": {
        "title": "🌿 Cotton Leaf Disease AI Diagnosis",
        "subtitle": "Clinical two-stage deep learning pipeline for cotton leaf validation, 9-class pathology diagnosis, and field prescriptions.",
        "input_header": "📥 Input Mode",
        "tab_upload": "📁 Single Leaf Upload",
        "tab_camera": "📷 Live Camera Capture",
        "tab_samples": "🖼️ Built-in Samples",
        "tab_survey": "🌾 Field Survey (Batch Scan)",
        "gatekeeper_title": "🛡️ Gatekeeper Verification",
        "verified_cotton": "✅ Verified Cotton Leaf",
        "not_cotton": "⚠️ Non-Cotton Object / Leaf Detected",
        "diagnosis_title": "🩺 Disease Diagnosis",
        "confidence": "Model Confidence",
        "severity": "Severity Level",
        "pathogen": "Pathogen / Cause",
        "category": "Classification Category",
        "symptoms_tab": "🔍 Symptoms",
        "actions_tab": "⚡ Immediate Actions (24-48h)",
        "chemical_tab": "💊 Chemical Treatment & Dosage",
        "organic_tab": "🌱 Organic & Biological",
        "prevention_tab": "🛡️ Prevention & Weather",
        "differential_title": "📊 Differential Diagnosis (Top Candidates)",
        "heatmap_title": "🔥 Lesion Focus Heatmap",
        "report_pdf_btn": "📄 Download Clinical Prescription (PDF)",
        "report_txt_btn": "📝 Download Text Summary",
        "ask_doctor_title": "💬 Ask Crop Doctor (AI Agronomist)",
        "survey_title": "🌾 Multi-Leaf Field Survey Analysis",
        "survey_summary": "Whole-Field Pathology Summary",
        "disclaimer": "⚠️ This AI decision support tool assists farmers and agronomists. For large acreage spraying, consult your local Krishi Vigyan Kendra (KVK) or extension specialist."
    },
    "hi": {
        "title": "🌿 कपास पत्ती रोग पहचान एवं निदान प्रणाली",
        "subtitle": "कपास की पत्तियों के रोगों की सटीक पहचान, जैविक व रासायनिक उपचार, पीडीएफ पर्ची और विशेषज्ञ सलाह।",
        "input_header": "📥 पत्ती की तस्वीर प्रदान करें",
        "tab_upload": "📁 एक पत्ती अपलोड करें",
        "tab_camera": "📷 कैमरा से खींचें",
        "tab_samples": "🖼️ उदाहरण फोटो",
        "tab_survey": "🌾 खेत सर्वेक्षण (बैच स्कैन)",
        "gatekeeper_title": "🛡️ पत्ती सत्यापन (गेटकीपर)",
        "verified_cotton": "✅ कपास की पत्ती सत्यापित",
        "not_cotton": "⚠️ यह कपास की पत्ती प्रतीत नहीं होती",
        "diagnosis_title": "🩺 रोग का निदान",
        "confidence": "सटीकता (Confidence)",
        "severity": "गंभीरता का स्तर",
        "pathogen": "रोगजनक (कारण)",
        "category": "रोग की श्रेणी",
        "symptoms_tab": "🔍 मुख्य लक्षण",
        "actions_tab": "⚡ आपातकालीन कदम (24-48 घंटे)",
        "chemical_tab": "💊 रासायनिक दवा व छिड़काव मात्रा",
        "organic_tab": "🌱 जैविक उपचार व देसी नुस्खे",
        "prevention_tab": "🛡️ बचाव व अनुकूल मौसम",
        "differential_title": "📊 संभावित अन्य रोग (Differential)",
        "heatmap_title": "🔥 प्रभावित क्षेत्र (हीटमैप)",
        "report_pdf_btn": "📄 डॉक्टर पर्ची डाउनलोड करें (PDF)",
        "report_txt_btn": "📝 टेक्स्ट रिपोर्ट डाउनलोड करें",
        "ask_doctor_title": "💬 फसल डॉक्टर से पूछें (AI Agronomist)",
        "survey_title": "🌾 संपूर्ण खेत सर्वेक्षण विश्लेषण",
        "survey_summary": "पूरे खेत की फसल स्वास्थ्य स्थिति",
        "disclaimer": "⚠️ यह एआई प्रणाली किसान सहायता हेतु है। बड़े पैमाने पर कीटनाशक छिड़काव से पूर्व निकटतम कृषि विज्ञान केंद्र से संपर्क करें।"
    },
    "mr": {
        "title": "🌿 कापूस पान रोग निदान आणि उपचार प्रणाली",
        "subtitle": "कपाशीच्या पानांवरील रोगांची खात्रीशीर तपासणी, जैविक व रासायनिक उपाययोजना, पीडीएफ रिपोर्ट आणि सल्ला.",
        "input_header": "📥 फोटो पर्याय निवडा",
        "tab_upload": "📁 एक पान अपलोड करा",
        "tab_camera": "📷 कॅमेऱ्याने फोटो काढा",
        "tab_samples": "🖼️ नमुना फोटो",
        "tab_survey": "🌾 शेत सर्वेक्षण (अनेक पाने)",
        "gatekeeper_title": "🛡️ पान पडताळणी",
        "verified_cotton": "✅ कपाशीचे पान खात्रीशीर",
        "not_cotton": "⚠️ हे कपाशीचे पान दिसत नाही",
        "diagnosis_title": "🩺 रोगाचे निदान",
        "confidence": "विश्वासार्हता (Confidence)",
        "severity": "तीव्रता",
        "pathogen": "रोगाचे कारण / विषाणू",
        "category": "रोगाचा प्रकार",
        "symptoms_tab": "🔍 लक्षणे",
        "actions_tab": "⚡ तात्काळ उपाय (२४-४८ तास)",
        "chemical_tab": "💊 रासायनिक औषधे आणि प्रमाण",
        "organic_tab": "🌱 सेंद्रिय व जैविक नियंत्रण",
        "prevention_tab": "🛡️ प्रतिबंधात्मक काळजी",
        "differential_title": "📊 इतर संभाव्य रोग",
        "heatmap_title": "🔥 रोगाचा प्रादुर्भाव क्षेत्र",
        "report_pdf_btn": "📄 कृषी प्रिस्क्रिप्शन डाउनलोड करा (PDF)",
        "report_txt_btn": "📝 माहिती रिपोर्ट",
        "ask_doctor_title": "💬 पीक सल्लागाराला विचारा (AI Agronomist)",
        "survey_title": "🌾 शेत सर्वेक्षण निष्कर्ष",
        "survey_summary": "संपूर्ण शेतातील रोगाची स्थिती",
        "disclaimer": "⚠️ ही प्रणाली शेतकरी मदतीसाठी आहे. फवारणीपूर्वी स्थानिक कृषी तज्ञांचा सल्ला घ्या."
    },
    "gu": {
        "title": "🌿 કપાસના પાન ના રોગોની ઓળખ અને સચોટ નિદાન",
        "subtitle": "કપાસના પાન ના રોગોનું સચોટ પરીક્ષણ, રાસાયણિક અને જૈવિક ઉપચાર માર્ગદર્શન, પીડીએફ રિપોર્ટ.",
        "input_header": "📥 ફોટો દાખલ કરો",
        "tab_upload": "📁 સિંગલ પાન અપલોડ કરો",
        "tab_camera": "📷 કેમેરા થી ફોટો પાડો",
        "tab_samples": "🖼️ નમૂના ફોટા",
        "tab_survey": "🌾 ખેતર સર્વે (એકસાથે તપાસ)",
        "gatekeeper_title": "🛡️ પાન ની ચકાસણી",
        "verified_cotton": "✅ કપાસનું પાન સાબિત થયું",
        "not_cotton": "⚠️ આ કપાસનું પાન લાગતું નથી",
        "diagnosis_title": "🩺 રોગનું નિદાન",
        "confidence": "ચોકસાઈ (Confidence)",
        "severity": "ગંભીરતા",
        "pathogen": "રોગ નું કારણ",
        "category": "રોગ નો પ્રકાર",
        "symptoms_tab": "🔍 મુખ્ય લક્ષણો",
        "actions_tab": "⚡ તાત્કાલિક પગલાં (૨૪-૪૮ કલાક)",
        "chemical_tab": "💊 દવા અને છંટકાવનું માપ",
        "organic_tab": "🌱 જૈવિક અને દેશી ઉપચાર",
        "prevention_tab": "🛡️ આગોતરી કાળજી",
        "differential_title": "📊 અન્ય સંભવિત રોગો",
        "heatmap_title": "🔥 અસરગ્રસ્ત ભાગ (હીટમેપ)",
        "report_pdf_btn": "📄 નિદાન રિપોર્ટ ડાઉનલોડ કરો (PDF)",
        "report_txt_btn": "📝 લખાણ રિપોર્ટ",
        "ask_doctor_title": "💬 પાક ડોક્ટર ને પૂછો (AI Agronomist)",
        "survey_title": "🌾 સમગ્ર ખેતર સર્વેક્ષણ",
        "survey_summary": "આખા ખેતરની આરોગ્ય સ્થિતિ",
        "disclaimer": "⚠️ આ એઆઈ સિસ્ટમ ખેડૂત મિત્રોના માર્ગદર્શન માટે છે. મોટા પાયે દવાનો છંટકાવ કરતા પહેલા કૃષિ નિષ્ણાતની સલાહ લો."
    }
}


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.title(":material/eco: Cotton care")
    st.caption("Your field companion")
    st.subheader("Preferences")
    st.caption("Light / dark theme: open the top-right menu → Settings → Theme.")

    lang_choice = st.selectbox(
        "🌐 Language / भाषा / ભાષા",
        options=["English", "हिंदी (Hindi)", "मराठी (Marathi)", "ગુજરાતી (Gujarati)"],
        index=0
    )
    lang_code = "en"
    if "हिंदी" in lang_choice:
        lang_code = "hi"
    elif "मराठी" in lang_choice:
        lang_code = "mr"
    elif "ગુજરાતી" in lang_choice:
        lang_code = "gu"

    T = UI_TEXTS[lang_code]

    st.space("small")
    st.subheader("⚙️ Analysis Pipeline")
    enforce_gatekeeper = st.toggle("Enforce Cotton Leaf Gatekeeper", value=True, help="Rejects non-cotton leaves, backgrounds, and arbitrary objects before disease diagnosis.")
    show_heatmap = st.toggle("Show color anomaly overlay", value=True, help="A color-and-edge visualization; it does not explain the model prediction.")
    top_k = st.slider("Differential Candidates to Show", min_value=2, max_value=5, value=3)

    st.space("small")
    with st.expander("📚 Supported 9 Disease Classes"):
        for name in engine.class_names:
            st.markdown(f"• **{name}**")


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.caption("COTTON CARE  /  LEAF HEALTH WORKSPACE")
st.title("A clearer picture of your crop." if lang_code == "en" else T["title"])
st.write("Inspect a leaf, understand possible conditions, and plan your next steps." if lang_code == "en" else T["subtitle"])
with st.container(horizontal=True):
    st.badge("Cotton verification", icon=":material/verified:", color="green")
    st.badge("9 leaf conditions", color="green")
    st.badge("4 languages", color="gray")
st.subheader("Start a leaf check")
st.caption("Use a sharp photo in natural light. Keep one leaf in focus and fill the frame.")


# ---------------------------------------------------------
# INPUT TABS (Single Upload, Camera, Samples, Field Survey)
# ---------------------------------------------------------
selected_image = None
image_source_label = ""
batch_mode = False
batch_images = []

input_mode = st.radio(
    "Image source", [T["tab_upload"], T["tab_camera"], T["tab_samples"], T["tab_survey"]],
    horizontal=True, key="input_mode"
)

if input_mode == T["tab_upload"]:
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image of a cotton leaf:",
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )
    if uploaded_file is not None:
        try:
            selected_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            image_source_label = f"Uploaded: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error opening image: {e}")

if input_mode == T["tab_camera"]:
    camera_photo = st.camera_input("Take a photo of the cotton leaf:", key="camera")
    if camera_photo is not None:
        try:
            selected_image = ImageOps.exif_transpose(Image.open(camera_photo)).convert("RGB")
            image_source_label = "Live Camera Capture"
        except Exception as e:
            st.error(f"Error reading camera photo: {e}")

if input_mode == T["tab_samples"]:
    st.write("Click any sample image below to run an instant diagnosis:")
    sample_dir = os.path.join(engine.base_dir, "sample_images")
    if os.path.exists(sample_dir):
        col1, col2, col3 = st.columns(3)
        sample_files = [
            ("healthy_leaf.jpg", "Healthy Cotton Leaf", col1),
            ("bacterial_blight.jpg", "Bacterial Blight Leaf", col2),
            ("non_cotton_ficus.jpg", "Non-Cotton Plant (Gatekeeper Test)", col3)
        ]
        for fname, label, col in sample_files:
            fpath = os.path.join(sample_dir, fname)
            if os.path.exists(fpath):
                thumb = Image.open(fpath)
                col.image(thumb, caption=label, width="stretch")
                if col.button(f"Load {label.split()[0]}", key=f"btn_{fname}"):
                    st.session_state["sample_path"] = fpath
                    st.session_state["sample_label"] = f"Sample: {label}"
        if st.session_state.get("sample_path"):
            selected_image = Image.open(st.session_state["sample_path"]).convert("RGB")
            image_source_label = st.session_state["sample_label"]

if input_mode == T["tab_survey"]:
    st.markdown("### 🌾 Multi-Leaf Field Survey Mode")
    st.write("Upload 2 to 10 leaf photos from different plants across your field to assess the total crop disease incidence rate.")
    survey_files = st.file_uploader(
        "Upload multiple cotton leaf photos:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="survey_uploader"
    )
    if survey_files and not 2 <= len(survey_files) <= 10:
        st.warning("Please select between 2 and 10 photos.")
    elif survey_files:
        batch_mode = True
        batch_images = survey_files


# ---------------------------------------------------------
# FIELD SURVEY BATCH PROCESSING
# ---------------------------------------------------------
if batch_mode and batch_images:
    st.space("small")
    st.subheader(T["survey_title"])
    st.write(f"Analyzing **{len(batch_images)} field specimens**...")

    survey_results = []
    disease_counter = {}

    survey_cols = st.columns(min(4, len(batch_images)))
    for idx, f in enumerate(batch_images):
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            res = engine.analyze(im, enforce_cotton_check=enforce_gatekeeper)
            res["filename"] = f.name
            res["image"] = im
            survey_results.append(res)

            pred = res["predicted_class"]
            if res["is_cotton"]:
                disease_counter[pred] = disease_counter.get(pred, 0) + 1

            col = survey_cols[idx % len(survey_cols)]
            with col:
                st.image(im, caption=f"#{idx+1}: {f.name[:14]}", width="stretch")
                badge_bg = "#4CAF50" if pred == "Healthy Leaf" else "#d32f2f"
                st.badge(pred, color="green" if pred == "Healthy Leaf" else "orange")
                if res.get("confidence"):
                    st.caption(f"Conf: {res['confidence']:.1f}%")
        except Exception as e:
            st.warning(f"Could not analyze {f.name}: {e}")

    total_scanned = len(survey_results)
    verified_count = sum(disease_counter.values())
    healthy_count = disease_counter.get("Healthy Leaf", 0)
    flagged_count = verified_count - healthy_count
    st.subheader("Sample summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Verified cotton leaves", verified_count, border=True)
    c2.metric("Flagged conditions", f"{flagged_count / verified_count:.0%}" if verified_count else "N/A", border=True)
    c3.metric("Excluded images", total_scanned - verified_count, border=True)
    if verified_count:
        st.caption("Results describe only these sampled leaves. Flagged conditions include non-infectious damage; this is not a field-wide infection estimate.")
        st.bar_chart(disease_counter, color="#267158")
    else:
        st.warning("No verified cotton leaves were available. Try clearer cotton leaf photos.")


# ---------------------------------------------------------
# SINGLE SPECIMEN ANALYSIS & CLINICAL DIAGNOSIS DISPLAY
# ---------------------------------------------------------
if selected_image is not None and not batch_mode:
    st.space("small")
    
    with st.spinner("Analyzing image with dual-stage AI pipeline..."):
        try:
            analysis = engine.analyze(selected_image, enforce_cotton_check=enforce_gatekeeper)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.stop()

    col_img, col_diag = st.columns([1, 1.4], gap="large")

    with col_img:
        st.subheader("🖼️ Inspected Specimen")
        st.image(selected_image, caption=image_source_label, width="stretch")

        if show_heatmap and analysis.get("heatmap") is not None:
            st.markdown(f"#### {T['heatmap_title']}")
            st.image(analysis["heatmap"], caption="Color and edge overlay — not model attention or a lesion measurement", width="stretch")

    with col_diag:
        is_cotton = analysis.get("is_cotton", True)
        gate_score = analysis.get("gate_score", 0.0)

        # 1. GATEKEEPER VERIFICATION BADGE
        st.markdown(f"### {T['gatekeeper_title']}")
        if is_cotton:
            st.success(f"{T['verified_cotton']} (Gate Score: {1.0 - gate_score:.1%})")
        else:
            st.error(f"{T['not_cotton']}")
            st.warning(analysis.get("message", "Image does not match known cotton leaf morphology."))
            if enforce_gatekeeper:
                st.info("💡 To inspect anyway, turn off **'Enforce Cotton Leaf Gatekeeper'** in the sidebar.")

        # 2. DISEASE DIAGNOSIS
        if is_cotton or not enforce_gatekeeper:
            pred_class = analysis["predicted_class"]
            confidence = analysis["confidence"]
            info = analysis.get("disease_info", {})

            # Multi-lingual translation lookup
            display_name = pred_class
            quick_advice = ""
            if lang_code != "en" and "translations" in info and lang_code in info["translations"]:
                trans = info["translations"][lang_code]
                display_name = trans.get("name", pred_class)
                quick_advice = trans.get("quick_advice", "")

            sev_level = info.get("severity", "Moderate")
            sev_color = info.get("severity_color", "#FFA500")

            st.markdown(f"### {T['diagnosis_title']}")
            with st.container(border=True):
                st.badge(sev_level + " severity", color="green" if pred_class == "Healthy Leaf" else "orange")
                st.subheader(display_name)
                st.caption(f"{info.get('category', 'Condition')} · {info.get('pathogen', 'N/A')}")

            c1, c2 = st.columns(2)
            c1.metric(T["confidence"], f"{confidence:.2f}%")
            c2.metric(T["severity"], sev_level)

            st.progress(min(1.0, max(0.0, confidence / 100.0)))

            if analysis.get("is_low_confidence"):
                st.warning("⚠️ Confidence is below 60%. Consider retaking the photo with sharper focus and balanced natural lighting.")
            elif analysis.get("is_ambiguous"):
                st.info(f"ℹ️ Close secondary candidate: **{analysis['second_class']}** ({analysis['second_confidence']:.1f}%). Observe multi-symptom characteristics.")

            if quick_advice:
                st.info(f"💡 **तात्काळ सल्ला / સલાહ**: {quick_advice}")

            # 3. STRUCTURED CLINICAL TREATMENT TABS
            tab_sympt, tab_act, tab_chem, tab_org, tab_prev = st.tabs([
                T["symptoms_tab"],
                T["actions_tab"],
                T["chemical_tab"],
                T["organic_tab"],
                T["prevention_tab"]
            ])

            with tab_sympt:
                st.markdown(f"**Overview:** {info.get('overview', '')}")
                st.markdown("#### Characteristic Indicators:")
                for s in info.get("symptoms", []):
                    st.markdown(f"• {s}")

            with tab_act:
                st.markdown("#### Immediate Containment Protocol (First 24-48 Hours):")
                for a in info.get("immediate_actions", []):
                    st.markdown(f"• {a}")

            with tab_chem:
                st.markdown("#### Approved Chemical Formulations & Dosage:")
                for c in info.get("chemical_control", []):
                    st.markdown(f"• {c}")

            with tab_org:
                st.markdown("#### Biological & Organic Alternatives:")
                for o in info.get("organic_biological", []):
                    st.markdown(f"• {o}")

            with tab_prev:
                st.markdown("#### Preventive Cultural Practices:")
                for p in info.get("prevention", []):
                    st.markdown(f"• {p}")
                if "favorable_weather" in info:
                    st.markdown(f"**Weather Triggers:** {info['favorable_weather']}")

            # 4. DIFFERENTIAL DIAGNOSIS BREAKDOWN
            st.markdown(f"### {T['differential_title']}")
            for candidate in analysis.get("all_predictions", [])[:top_k]:
                c_name = candidate["class_name"]
                c_prob = candidate["probability"]
                st.write(f"**{c_name}**: {c_prob:.2f}%")
                st.progress(c_prob / 100.0)

            # 5. DUAL DOWNLOAD PRESCRIPTION REPORTS (PDF + TXT)
            st.space("small")
            st.markdown("#### 📄 Agronomic Prescription Downloads")

            # Generate PDF in-memory
            pdf_bytes = generate_pdf_report(
                analysis_result=analysis,
                original_image=selected_image,
                specimen_label=image_source_label
            )

            # Generate Text summary
            report_text = f"""=====================================================
COTTON LEAF PATHOLOGY & AGRONOMIC DIAGNOSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================================
SPECIMEN: {image_source_label}
DIAGNOSIS: {pred_class} (Confidence: {confidence:.2f}%)
SEVERITY: {sev_level}
PATHOGEN: {info.get('pathogen', 'N/A')}

IMMEDIATE ACTIONS:
""" + "\n".join([f"- {a}" for a in info.get("immediate_actions", [])]) + """

RECOMMENDED CHEMICAL CONTROL:
""" + "\n".join([f"- {c}" for c in info.get("chemical_control", [])]) + """

ORGANIC MANAGEMENT:
""" + "\n".join([f"- {o}" for o in info.get("organic_biological", [])])

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label=T["report_pdf_btn"],
                    data=pdf_bytes,
                    file_name=f"cotton_prescription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            with btn_col2:
                st.download_button(
                    label=T["report_txt_btn"],
                    data=report_text,
                    file_name=f"cotton_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )


# ---------------------------------------------------------
# INTERACTIVE AI AGRONOMIST (CROP DOCTOR CHAT)
# ---------------------------------------------------------
st.space("small")
st.markdown(f"### {T['ask_doctor_title']}")
st.write("Ask practical questions regarding spray dosages, tank mixing compatibility, rain precautions, or organic treatments:")

# Quick query chips
chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
user_query = None

if chip_col1.button("🧪 Tank Mix Safety", key="chip_mix"):
    user_query = "Can I mix Mancozeb with other chemicals in a spray tank?"
if chip_col2.button("🌧️ Spray in Rain / Wet Leaves", key="chip_rain"):
    user_query = "How many hours after spray before it rains?"
if chip_col3.button("🪰 Whitefly & Curl Virus", key="chip_fly"):
    user_query = "How to control whitefly vector for leaf curl virus?"
if chip_col4.button("🍂 Lal Patti (Leaf Redding)", key="chip_red"):
    user_query = "How to treat leaf reddening lal patti in cotton?"

custom_query = st.text_input("Or type your own farming question here:", value=user_query if user_query else "", placeholder="e.g. Can I mix Urea with Magnesium Sulphate?")

if custom_query:
    current_diag = analysis.get("predicted_class") if selected_image is not None and 'analysis' in locals() else None
    current_info = engine.disease_info
    
    with st.spinner("Consulting Crop Doctor knowledge base..."):
        answer = ask_agronomist(custom_query, current_disease=current_diag, disease_info=current_info)

    with st.container(border=True):
        st.write(custom_query)
        st.markdown(answer)


# ---------------------------------------------------------
# FOOTER & DISCLAIMER
# ---------------------------------------------------------
st.space("small")
st.caption(T["disclaimer"])
