import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

# ==========================
# Konfigurasi halaman
# ==========================
st.set_page_config(page_title="Dashboard Model - Balqis Isaura", layout="wide")
st.title("🎯 Dashboard Model - Balqis Isaura")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("model/mask_detection.pt")  # ganti dengan path YOLO kamu
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        return None

@st.cache_resource
def load_resnet_model():
    try:
        model = tf.keras.models.load_model("model/rps_classifier.h5")  # ganti dengan model kamu
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model TensorFlow: {e}")
        return None


# ==========================
# Pilihan Model
# ==========================
st.sidebar.header("Pilih Model:")
model_option = st.sidebar.radio(
    "Model yang digunakan:",
    ("PyTorch - YOLO", "TensorFlow - ResNet50")
)

uploaded_file = st.file_uploader("📤 Upload gambar untuk deteksi / klasifikasi:", type=["jpg", "png", "jpeg"])

# ==========================
# Proses YOLO (Mask Detection)
# ==========================
if model_option == "PyTorch - YOLO":
    model = load_yolo_model()
    st.subheader("🧠 Model Deteksi Objek (Mask Detection)")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Gambar Asli", use_container_width=True)

        image_data = Image.open(uploaded_file)
        results = None

        if st.button("▶ Jalankan Deteksi YOLO"):
            with st.spinner("Mendeteksi objek..."):
                results = model(image_data)
            
            res_plotted = results[0].plot()[:, :, ::-1]
            with col2:
                st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)

            # Cek apakah objek terdeteksi
            if len(results[0].boxes) == 0:
                st.warning("⚠️ Tidak ada objek yang terdeteksi.")
            else:
                st.success("✅ Objek berhasil terdeteksi!")
                st.write("📄 Detail deteksi:")
                st.dataframe(results[0].boxes.data.cpu().numpy())
    else:
        st.info("⬆️ Silakan upload gambar untuk memulai deteksi.")


# ==========================
# Proses TensorFlow (RPS Classification)
# ==========================
elif model_option == "TensorFlow - ResNet50":
    model = load_resnet_model()
    st.subheader("✋ Model Klasifikasi Rock-Paper-Scissors")

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Gambar Asli", use_container_width=True)

        if st.button("▶ Jalankan Klasifikasi RPS"):
            try:
                # Ambil input shape model (misal (150,150,3) atau (224,224,3))
                input_shape = model.input_shape[1:3]

                # Preprocessing gambar
                img = Image.open(uploaded_file).convert("RGB")
                img_resized = img.resize(input_shape)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension
                img_array = img_array / 255.0  # Normalisasi

                # Prediksi
                preds = model.predict(img_array)
                class_names = ["Rock", "Paper", "Scissors"]
                predicted_class = class_names[np.argmax(preds)]
                confidence = np.max(preds)

                with col2:
                    st.success(f"🧩 Hasil Prediksi: **{predicted_class}**")
                    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

            except Exception as e:
                st.error(f"❌ Error saat klasifikasi: {e}")
    else:
        st.info("⬆️ Upload gambar untuk mulai klasifikasi.")
