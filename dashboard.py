import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os

# ===================== KONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="💻",
    layout="wide"
)

# ===================== STYLE MODERN PURPLE-BLUE THEME =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    /* Main Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%) !important;
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Hide default padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, #f0f0f5 0%, #faf5ff 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .header-badge {
        display: inline-block;
        border: 2px solid #6366f1;
        color: #6366f1;
        padding: 8px 24px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 15px;
        letter-spacing: 1px;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 10px 0;
        background: linear-gradient(135deg, #00d4ff 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    
    .header-subtitle {
        font-size: 2rem;
        font-weight: 700;
        color: #7c3aed;
        margin-bottom: 25px;
    }
    
    .header-tabs {
        display: flex;
        gap: 15px;
        margin-top: 20px;
    }
    
    .header-tab {
        flex: 1;
        padding: 15px 30px;
        border: 2px solid #6366f1;
        border-radius: 25px;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
        color: #6366f1;
        background: white;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .laptop-illustration {
        position: absolute;
        right: 40px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 8rem;
        opacity: 0.9;
    }
    
    /* Grid Background for bottom section */
    .grid-background {
        background: linear-gradient(135deg, #00d4ff 0%, #6366f1 50%, #7c3aed 100%);
        padding: 60px 40px;
        margin: 30px -50px -50px -50px;
        position: relative;
        overflow: hidden;
    }
    
    .grid-background::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        transform: perspective(500px) rotateX(60deg);
        transform-origin: bottom;
    }
    
    /* Process Cards */
    .process-card {
        background: white;
        border-radius: 20px;
        padding: 35px;
        margin: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        position: relative;
        z-index: 1;
        min-height: 600px;
    }
    
    .process-card h2 {
        color: #6366f1 !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .process-number {
        display: inline-block;
        background: linear-gradient(135deg, #00d4ff, #6366f1);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: 700;
        margin-right: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #6366f1) !important;
        color: white !important;
        border: none !important;
        padding: 14px 32px !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f0f9ff, #f5f3ff);
        border: 2px dashed #6366f1;
        border-radius: 15px;
        padding: 25px;
    }
    
    [data-testid="stFileUploader"] label {
        color: #6366f1 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #6366f1 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #6366f1) !important;
        border-radius: 10px;
    }
    
    /* Images */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Success/Info/Error Messages */
    .stSuccess, .stInfo, .stError {
        border-radius: 12px;
        padding: 15px;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f9ff, #f5f3ff);
        border-radius: 10px;
        font-weight: 600;
        color: #6366f1;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6366f1, #7c3aed) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-title { font-size: 2rem; }
        .header-subtitle { font-size: 1.2rem; }
        .laptop-illustration { display: none; }
    }
    </style>
""", unsafe_allow_html=True)

# ===================== HEADER SECTION =====================
st.markdown("""
    <div class="header-container">
        <div class="laptop-illustration">💻</div>
        <div class="header-badge">BY BALQIS ISAURA</div>
        <div class="header-title">DASHBOARD</div>
        <div class="header-subtitle">DETEKSI OBJEK & KLASIFIKASI GAMBAR</div>
        <div class="header-tabs">
            <div class="header-tab">DETEKSI OBJEK</div>
            <div class="header-tab">KLASIFIKASI GAMBAR</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ===================== GRID BACKGROUND SECTION WITH TWO COLUMNS =====================
st.markdown('<div class="grid-background">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

# ===================== PROCESS 1 - YOLO OBJECT DETECTION =====================
with col1:
    st.markdown("""
        <div class="process-card">
            <h2><span class="process-number">1</span>DETEKSI OBJEK - YOLO</h2>
    """, unsafe_allow_html=True)
    
    try:
        @st.cache_resource
        def load_yolo():
            return YOLO("model/Balqis Isaura_Laporan 4.pt")

        with st.spinner("🔄 Memuat model YOLO..."):
            yolo_model = load_yolo()
        st.success("✅ Model YOLO berhasil dimuat!")

        uploaded_file_yolo = st.file_uploader(
            "📁 Upload gambar untuk deteksi objek:", 
            type=["jpg", "jpeg", "png"], 
            key="yolo"
        )

        if uploaded_file_yolo:
            image = Image.open(uploaded_file_yolo)
            st.image(image, caption="📷 Gambar Input", use_container_width=True)

            if st.button("🚀 Jalankan Deteksi", key="detect_btn"):
                with st.spinner("🔍 Mendeteksi objek..."):
                    results = yolo_model(image)
                    result_img = results[0].plot()
                    st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

                    st.markdown("#### 📋 Detail Deteksi")
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes, 1):
                            label = yolo_model.names[int(box.cls)]
                            conf = box.conf[0]
                            st.write(f"**{i}.** {label} — Confidence: **{conf:.2%}**")
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi.")
                        
    except Exception as e:
        st.error(f"❌ Error YOLO: {e}")
        st.info("💡 Pastikan file `.pt` ada di folder `model/`.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== PROCESS 2 - TENSORFLOW CLASSIFICATION =====================
with col2:
    st.markdown("""
        <div class="process-card">
            <h2><span class="process-number">2</span>KLASIFIKASI - TENSORFLOW</h2>
    """, unsafe_allow_html=True)
    
    try:
        FILE_ID = "1uYmpPANnUKNKBaRHCOlylWV7t3fDgPp2"
        MODEL_PATH = "model_resnet50_balqis.h5"

        if not os.path.exists(MODEL_PATH):
            with st.spinner("⬇️ Mengunduh model dari Google Drive..."):
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("✅ Model berhasil diunduh!")

        @st.cache_resource
        def load_tf_model():
            return tf.keras.models.load_model(MODEL_PATH, compile=False)

        with st.spinner("🔄 Memuat model TensorFlow..."):
            model = load_tf_model()
        st.success("✅ Model TensorFlow berhasil dimuat!")

        class_names = ["Rock", "Paper", "Scissors"]

        uploaded_file_tf = st.file_uploader(
            "📁 Upload gambar untuk klasifikasi:", 
            type=["jpg", "jpeg", "png"], 
            key="tf"
        )

        if uploaded_file_tf:
            image = Image.open(uploaded_file_tf)
            st.image(image, caption="📷 Gambar Input", use_container_width=True)

            if st.button("🔮 Prediksi Gambar", key="predict_btn"):
                with st.spinner("🤖 Melakukan prediksi..."):
                    img_array = np.array(image.resize((224, 224))) / 255.0
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array]*3, axis=-1)
                    elif img_array.shape[-1] == 4:
                        img_array = img_array[..., :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_index]
                    confidence = predictions[0][predicted_index]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🎯 Kelas Prediksi", predicted_class)
                    with col_b:
                        st.metric("📊 Confidence", f"{confidence:.2%}")

                    with st.expander("📊 Lihat Probabilitas Tiap Kelas"):
                        for i, prob in enumerate(predictions[0]):
                            st.progress(float(prob), text=f"**{class_names[i]}**: {prob:.4f}")
                            
    except Exception as e:
        st.error(f"❌ Error TensorFlow: {str(e)}")
        st.info("""
        💡 **Tips:**
        - Pastikan link Google Drive publik
        - Model harus memiliki 3 output kelas: Rock, Paper, Scissors
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
    <div style="text-align: center; padding: 30px; color: #666; font-weight: 600;">
        💙 Dibuat oleh <strong style="color: #6366f1;">Balqis Isaura</strong> | Powered by Streamlit
    </div>
""", unsafe_allow_html=True)
