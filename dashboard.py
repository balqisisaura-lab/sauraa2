import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ==========================
# Konfigurasi Dashboard
# ==========================
st.set_page_config(page_title="Dashboard AI", layout="wide")

# Judul dan Deskripsi
st.markdown("""
    <h1 style='text-align: center; color: #fff;'>💡 Dashboard AI - Deteksi & Klasifikasi Gambar</h1>
    <p style='text-align: center; color: #ccc;'>Coba fitur Deteksi Objek atau Klasifikasi Gambar menggunakan model AI kamu!</p>
""", unsafe_allow_html=True)

# Pilihan fitur
tab1, tab2 = st.tabs(["🔍 Deteksi Objek", "🧠 Klasifikasi Gambar"])

# ==========================
# 1️⃣ Deteksi Objek
# ==========================
with tab1:
    st.header("Deteksi Objek")
    uploaded_file = st.file_uploader("Upload Gambar disini 🖼️", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar diupload", use_column_width=True)

        # Load model YOLO (gunakan cache agar tidak load ulang tiap klik)
        @st.cache_resource
        def load_yolo():
            return YOLO("model/Balqis Isaura_Laporan 4.pt")
        
        yolo_model = load_yolo()

        # Deteksi objek
        st.write("🔎 Mendeteksi objek...")
        results = yolo_model(image)
        result_image = results[0].plot()  # hasil dengan bounding box
        st.image(result_image, caption="Hasil Deteksi", use_column_width=True)

# ==========================
# 2️⃣ Klasifikasi Gambar
# ==========================
with tab2:
    st.header("Klasifikasi Gambar")
    uploaded_file2 = st.file_uploader("Upload Gambar untuk diklasifikasi", type=["jpg", "jpeg", "png"], key="clasify")

    if uploaded_file2 is not None:
        image2 = Image.open(uploaded_file2).resize((224, 224))
        st.image(image2, caption="Gambar diupload", use_column_width=False)

        # Load model klasifikasi
        @st.cache_resource
        def load_classifier():
            return tf.keras.models.load_model("model/model_resnet50.keras")
        
        classifier = load_classifier()

        # Prediksi
        img_array = np.expand_dims(np.array(image2) / 255.0, axis=0)
        prediction = classifier.predict(img_array)

        # Kalau output 1 kelas (misal binary)
        if prediction.shape[1] == 1:
            kelas = "Positif" if prediction[0][0] > 0.5 else "Negatif"
            st.success(f"Hasil klasifikasi: **{kelas}** ({prediction[0][0]:.2f})")
        else:
            kelas = np.argmax(prediction)
            st.success(f"Hasil klasifikasi: Kelas **{kelas}**")
