import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

# ===================== STYLE SESUAI GAMBAR =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

body, [data-testid="stAppViewContainer"] {
    font-family: 'Poppins', sans-serif;
    background-color: #fffdfd;
}

#MainMenu, header, footer {visibility: hidden;}

.header-container {
    text-align: center;
    padding-top: 50px;
    padding-bottom: 50px;
    position: relative;
}

.header-badge {
    display: inline-block;
    border: 1.5px solid #9D8CFF;
    color: #7C4DFF;
    border-radius: 25px;
    padding: 6px 22px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

.header-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin: 15px 0 5px;
    color: #00C8FF;
}

.header-subtitle {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7C4DFF, #00C8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.button-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 30px;
}

.option-button {
    border: 1.5px solid #9D8CFF;
    background: white;
    color: #7C4DFF;
    border-radius: 25px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.option-button:hover {
    background: linear-gradient(90deg, #7C4DFF, #00C8FF);
    color: white;
}

.active {
    background: linear-gradient(90deg, #7C4DFF, #00C8FF);
    color: white;
}

.grid-bottom {
    margin-top: 70px;
    height: 120px;
    background: linear-gradient(90deg, #00C8FF, #7C4DFF);
    position: relative;
    overflow: hidden;
}

.grid-bottom::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
    background-size: 50px 50px;
    transform: perspective(400px) rotateX(60deg);
    transform-origin: bottom;
}

.process-card {
    background: white;
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER DAN TOMBOL PILIHAN =====================
st.markdown("""
<div class="header-container">
    <div class="header-badge">BY BALQIS ISAURA</div>
    <div class="header-title">DASHBOARD</div>
    <div class="header-subtitle">DETEKSI OBJEK & KLASIFIKASI GAMBAR</div>
</div>
""", unsafe_allow_html=True)

# Gunakan session state agar tombol bisa aktif sesuai pilihan
if "page" not in st.session_state:
    st.session_state.page = "deteksi"

col_btn1, col_btn2 = st.columns([1, 1], gap="medium")

with col_btn1:
    if st.button("🧠 DETEKSI OBJEK", use_container_width=True):
        st.session_state.page = "deteksi"
with col_btn2:
    if st.button("📸 KLASIFIKASI GAMBAR", use_container_width=True):
        st.session_state.page = "klasifikasi"

# Gaya aktif di CSS disimulasikan dengan markdown
st.markdown(
    f"""
    <style>
    button[kind="secondary"]:nth-child({1 if st.session_state.page == "deteksi" else 2}) {{
        background: linear-gradient(90deg, #7C4DFF, #00C8FF);
        color: white !important;
        border: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== HALAMAN 1: DETEKSI OBJEK =====================
if st.session_state.page == "deteksi":
    st.markdown('<div class="process-card">', unsafe_allow_html=True)
    st.subheader("🔍 DETEKSI OBJEK - YOLO")

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

            if st.button("🚀 Jalankan Deteksi"):
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

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== HALAMAN 2: KLASIFIKASI GAMBAR =====================
elif st.session_state.page == "klasifikasi":
    st.markdown('<div class="process-card">', unsafe_allow_html=True)
    st.subheader("🤖 KLASIFIKASI GAMBAR - TENSORFLOW")

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

            if st.button("🔮 Prediksi Gambar"):
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
        st.info("💡 Pastikan link Google Drive publik dan model punya 3 output kelas.")

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<div class="grid-bottom"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:20px; color:#666; font-weight:600;">
💙 Dibuat oleh <strong style="color:#7C4DFF;">Balqis Isaura</strong> | Powered by Streamlit
</div>
""", unsafe_allow_html=True), trus kasih sintaks lengkap yg udah diubah ke aku
