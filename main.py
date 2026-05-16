import streamlit as st
import tensorflow as tf
import numpy as np

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] * {color: #cbd5e1 !important;}

    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .sidebar-logo h1 {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .sidebar-logo p {
        font-size: 0.75rem;
        color: #64748b !important;
        margin-top: 0.2rem;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f4c2a 0%, #065f46 40%, #0e7490 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(74, 222, 128, 0.2);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(74,222,128,0.08) 0%, transparent 60%),
                    radial-gradient(circle at 70% 50%, rgba(34,211,238,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-banner h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.75);
        max-width: 600px;
        margin: 0 auto;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(74,222,128,0.15);
        border: 1px solid rgba(74,222,128,0.3);
        color: #4ade80;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        margin-bottom: 1rem;
        letter-spacing: 0.05em;
    }

    /* ── Stat Cards ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(74,222,128,0.3);
        transform: translateY(-2px);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* ── How It Works Cards ── */
    .steps-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .step-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        position: relative;
    }
    .step-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }
    .step-num {
        position: absolute;
        top: 1rem; right: 1rem;
        background: rgba(74,222,128,0.1);
        color: #4ade80;
        font-size: 0.7rem;
        font-weight: 700;
        width: 24px; height: 24px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        border: 1px solid rgba(74,222,128,0.2);
    }
    .step-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.4rem;
    }
    .step-desc {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.5;
    }

    /* ── Section Header ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Upload Zone ── */
    .upload-zone {
        background: rgba(74,222,128,0.03);
        border: 2px dashed rgba(74,222,128,0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(74,222,128,0.4);
        background: rgba(74,222,128,0.05);
    }

    /* ── Result Card ── */
    .result-card {
        background: linear-gradient(135deg, rgba(74,222,128,0.1), rgba(34,211,238,0.05));
        border: 1px solid rgba(74,222,128,0.25);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeInUp 0.5s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-plant {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4ade80;
        margin: 0.3rem 0;
    }
    .result-disease {
        font-size: 1.1rem;
        color: #94a3b8;
    }
    .result-healthy {
        color: #4ade80;
    }
    .result-diseased {
        color: #f87171;
    }

    /* ── About Cards ── */
    .about-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .about-card h3 {
        color: #4ade80;
        margin-top: 0;
        font-size: 1rem;
    }
    .data-pill {
        display: inline-block;
        background: rgba(34,211,238,0.1);
        border: 1px solid rgba(34,211,238,0.2);
        color: #22d3ee;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        margin: 0.2rem;
    }

    /* Streamlit widget tweaks */
    .stFileUploader {
        background: transparent !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(74,222,128,0.25) !important;
        border-radius: 16px !important;
        color: #94a3b8 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #16a34a, #0891b2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(22,163,74,0.4) !important;
    }
    [data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stImage img {
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }
    .stSpinner > div {color: #4ade80 !important;}
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        background: rgba(74,222,128,0.08) !important;
        border: 1px solid rgba(74,222,128,0.2) !important;
        color: #4ade80 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('trained_model.h5', compile=False)
    return model

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    return result_index, confidence

def format_result(class_name):
    """Split class name into plant and condition."""
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in condition.lower()
    return plant, condition, is_healthy


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>🌿 PlantGuard</h1>
        <p>AI-Powered Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    app_mode = st.selectbox(
        "Navigate",
        ["🏠  Home", "🔬  Disease Recognition", "📊  About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="padding: 0.5rem 0; font-size:0.8rem; color:#475569;">
        <p> 38 Plant Classes🌱</p>
        <p> Deep Learning Model📊</p>
        <p> Real-time Analysis⚡</p>
    </div>
    """, unsafe_allow_html=True)


# ── HOME PAGE ─────────────────────────────────────────────────────────────────
if "Home" in app_mode:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">✨ AI-Powered • Real-time • Accurate</div>
        <h1>🌿 PlantGuard AI</h1>
        <p>Detect plant diseases instantly with the power of deep learning. Upload a leaf image and get results in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-number">87K+</div>
            <div class="stat-label">Training Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">38</div>
            <div class="stat-label">Disease Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">98%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .hero-img-wrap { position:relative; border-radius:20px; overflow:hidden;
        box-shadow:0 12px 35px rgba(0,0,0,0.45); border:1px solid rgba(255,255,255,0.07);
        margin-bottom:0.5rem; }
    .hero-img-wrap img { height:220px !important; object-fit:cover; object-position:center 30%;
        border-radius:20px !important; width:100% !important; }
    .hero-img-wrap::after {
        content:"";
        position:absolute; bottom:0; left:0; right:0; height:60%;
        background:linear-gradient(to top, rgba(10,15,30,0.7) 0%, transparent 100%);
        border-radius:0 0 20px 20px; pointer-events:none;
    }
    </style>
    <div class="hero-img-wrap">
    """, unsafe_allow_html=True)

    st.image("home_page.jpeg", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-grid">
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-icon">📸</div>
            <div class="step-title">Upload a Leaf Image</div>
            <div class="step-desc">Take a clear photo of the affected plant leaf and upload it.</div>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-icon">🧠</div>
            <div class="step-title">AI Analysis</div>
            <div class="step-desc">Our CNN model analyzes the image across 38 disease categories.</div>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-icon">✅</div>
            <div class="step-title">Get Results</div>
            <div class="step-desc">Receive instant diagnosis with confidence score.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── DISEASE RECOGNITION PAGE ──────────────────────────────────────────────────
elif "Recognition" in app_mode:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h2 style="color:#f1f5f9; font-weight:700; margin:0;">🔬 Disease Recognition</h2>
        <p style="color:#64748b; margin-top:0.3rem;">Upload a plant leaf image to detect diseases instantly</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
        test_image = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if test_image:
            st.image(test_image, caption="Uploaded Image", use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍  Analyze Disease", use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size:3rem;">🌿</div>
                <p style="color:#64748b; margin:0.5rem 0 0 0;">Supported: JPG, PNG, WebP</p>
            </div>
            """, unsafe_allow_html=True)
            predict_btn = False

    with col_result:
        st.markdown('<div class="section-header">📋 Analysis Result</div>', unsafe_allow_html=True)

        if test_image and predict_btn:
            with st.spinner("🧠 Analyzing image..."):
                result_index, confidence = model_prediction(test_image)
                plant, condition, is_healthy = format_result(CLASS_NAMES[result_index])

            status_class = "result-healthy" if is_healthy else "result-diseased"
            status_icon = "✅" if is_healthy else "⚠️"
            status_text = "Healthy" if is_healthy else "Disease Detected"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">{status_icon}</div>
                <div style="font-size:0.85rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">Plant</div>
                <div class="result-plant">{plant}</div>
                <div style="font-size:0.85rem; color:#64748b; margin-top:0.8rem;">Condition</div>
                <div class="result-disease {status_class}">{condition}</div>
                <div style="margin-top:1.2rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:0.8rem; color:#64748b;">Confidence</div>
                    <div style="font-size:1.4rem; font-weight:700; color:#22d3ee;">{confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                        border-radius:20px; padding:3rem; text-align:center; height:300px;
                        display:flex; align-items:center; justify-content:center; flex-direction:column;">
                <div style="font-size:3rem; opacity:0.3;">🔬</div>
                <p style="color:#475569; margin:0.5rem 0 0 0;">Results will appear here</p>
            </div>
            """, unsafe_allow_html=True)


# ── ABOUT PAGE ────────────────────────────────────────────────────────────────
elif "About" in app_mode:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h2 style="color:#f1f5f9; font-weight:700; margin:0;">📊 About the Project</h2>
        <p style="color:#64748b; margin-top:0.3rem;">Dataset, model architecture, and technical details</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>🗃️ Dataset Overview</h3>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
                The PlantVillage dataset contains <strong style="color:#f1f5f9;">87,000+ RGB images</strong>
                of healthy and diseased crop leaves across 38 classes. Augmented offline for better
                generalization.
            </p>
            <div style="margin-top:1rem;">
                <span class="data-pill">📚 Train: 70,295</span>
                <span class="data-pill">✅ Valid: 17,572</span>
                <span class="data-pill">🧪 Test: 33</span>
            </div>
        </div>

        <div class="about-card">
            <h3>🤖 Model Architecture</h3>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
                Convolutional Neural Network (CNN) trained on 128×128 RGB images.
                Uses multiple Conv2D layers, MaxPooling, and Dense layers for classification.
            </p>
            <div style="margin-top:1rem;">
                <span class="data-pill">🧠 CNN</span>
                <span class="data-pill">📐 128×128</span>
                <span class="data-pill">🎯 38 Classes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <h3>🌱 Supported Plants</h3>
            <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.8rem;">
                The model can identify diseases in these crops:
            </p>
        """, unsafe_allow_html=True)

        plants = ["🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn (Maize)",
                  "🍇 Grape", "🍊 Orange", "🍑 Peach", "🫑 Bell Pepper",
                  "🥔 Potato", "🫐 Raspberry", "🌱 Soybean", "🎃 Squash",
                  "🍓 Strawberry", "🍅 Tomato"]

        pills_html = "".join([f'<span class="data-pill" style="border-color:rgba(74,222,128,0.2); color:#4ade80; background:rgba(74,222,128,0.08);">{p}</span>' for p in plants])
        st.markdown(f'<div>{pills_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card" style="margin-top:1rem;">
            <h3>📦 Tech Stack</h3>
            <div>
                <span class="data-pill">🐍 Python</span>
                <span class="data-pill">🧠 TensorFlow</span>
                <span class="data-pill">🎈 Streamlit</span>
                <span class="data-pill">🔢 NumPy</span>
                <span class="data-pill">🖼️ Pillow</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
