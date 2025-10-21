import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# Konfigurasi Dashboard
# ==========================
st.set_page_config(page_title="Dashboard Model - Balqis Isaura", page_icon="🎯", layout="centered")

st.title("🎯 Dashboard Model - Balqis Isaura")
st.caption("🪄 Upload gambar untuk deteksi / klasifikasi:")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/model_resnet50.keras")
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model TensorFlow: {e}")
        classifier = None

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📤 Upload gambar di sini", type=["jpg", "png", "jpeg"])

# ==========================
# Deteksi & Klasifikasi
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

    # Convert ke format yang bisa dibaca OpenCV
    img_array = np.array(img)

    st.subheader("📍 Pilih jenis prediksi:")
    mode = st.radio("Mode Prediksi", ["Deteksi Objek (YOLO)", "Klasifikasi (ResNet50)"], horizontal=True)

    if st.button("🚀 Jalankan Model"):
        with st.spinner("⏳ Sedang memproses..."):
            if mode == "Deteksi Objek (YOLO)":
                if yolo_model:
                    try:
                        results = yolo_model(img_array)
                        res_plotted = results[0].plot()
                        st.image(res_plotted, caption="📦 Hasil Deteksi Objek", use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat deteksi: {e}")
                else:
                    st.warning("⚠️ Model YOLO belum dimuat dengan benar.")

            elif mode == "Klasifikasi (ResNet50)":
                if classifier:
                    try:
                        img_resized = img.resize((224, 224))
                        x = image.img_to_array(img_resized)
                        x = np.expand_dims(x, axis=0)
                        x = x / 255.0

                        pred = classifier.predict(x)
                        kelas = np.argmax(pred, axis=1)[0]

                        st.success(f"✅ Hasil klasifikasi: **Kelas {kelas}**")
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
                else:
                    st.warning("⚠️ Model klasifikasi belum dimuat dengan benar.")
