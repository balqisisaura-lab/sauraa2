import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🤖",
    layout="wide",
)

# =========================
# Fungsi memuat model
# =========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/model_resnet50.keras", compile=False)
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model ResNet50: {e}")
        classifier = None

    return yolo_model, classifier


yolo_model, classifier = load_models()

# =========================
# Fungsi bantu
# =========================
def predict_class(img, model):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    return class_idx, confidence


def detect_objects(img, model):
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    labels = []
    for i in range(len(classes)):
        labels.append(f"{results[0].names[int(classes[i])]} ({confidences[i]:.2f})")
    return labels


# =========================
# Styling UI (CSS)
# =========================
st.markdown("""
<style>
body {
    background-color: #f5f5f7;
}
.title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2b2b2b;
    text-align: center;
}
.subtitle {
    font-size: 1rem;
    color: #666;
    text-align: center;
    margin-bottom: 30px;
}
.mode-select {
    background-color: #fff;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.result-card {
    background-color: #fff;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# UI Header
# =========================
st.markdown("<h1 class='title'>🤖 AI Vision Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Pilih mode di bawah untuk melakukan deteksi objek atau klasifikasi gambar.</p>", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Mode Analisis")
mode = st.sidebar.radio("Pilih Mode:", ["Klasifikasi Gambar", "Deteksi Objek"])

st.sidebar.write("---")
st.sidebar.info("Upload gambar untuk melihat hasil prediksi model AI.")

# =========================
# Upload gambar
# =========================
uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

    with st.spinner("🔍 Menganalisis gambar..."):
        if mode == "Klasifikasi Gambar":
            if classifier is not None:
                idx, conf = predict_class(img, classifier)
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(f"### 🎯 Hasil Klasifikasi: **Kelas {idx}**")
                st.markdown(f"**Confidence:** {conf:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Model klasifikasi belum dimuat.")
        else:
            if yolo_model is not None:
                labels = detect_objects(img, yolo_model)
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                if labels:
                    st.markdown("### 🧠 Hasil Deteksi Objek:")
                    for label in labels:
                        st.markdown(f"- {label}")
                else:
                    st.info("Tidak ada objek terdeteksi.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Model YOLO belum dimuat.")
else:
    st.info("📂 Silakan unggah gambar terlebih dahulu.")

# =========================
# Footer
# =========================
st.markdown("""
---
<div style='text-align:center; color:#aaa; font-size:0.9rem;'>
Developed by <b>Balqis Isaura</b> · Powered by TensorFlow & YOLO
</div>
""", unsafe_allow_html=True)
