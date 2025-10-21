import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import cv2

# ==========================
# Sidebar Pilihan Model
# ==========================
st.sidebar.title("Pilih Model:")
model_choice = st.sidebar.radio(
    "Model yang ingin digunakan:",
    ["PyTorch - YOLO (Mask Detection)", "TensorFlow - ResNet50 (Rock Paper Scissors)"]
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_yolo_model():
    return YOLO("model/Balqis Isaura_Laporan 4.pt")  # ganti sesuai nama file YOLO kamu

@st.cache_resource
def load_resnet_model():
    return tf.keras.models.load_model("model/model_resnet50.keras")  # ganti sesuai nama file H5 kamu

# ==========================
# Upload Gambar
# ==========================
st.title("🧠 Dashboard Deteksi & Klasifikasi Gambar")

uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    # ==========================
    # YOLO - Deteksi Masker
    # ==========================
    if model_choice.startswith("PyTorch"):
        model = load_yolo_model()
        st.subheader("🔍 Jalankan Deteksi YOLO")

        if st.button("Jalankan Deteksi"):
            results = model.predict(source=img, conf=0.5, verbose=False)
            result_img = results[0].plot()  # hasil dengan bounding box
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

            # Menampilkan label hasil deteksi
            if len(results[0].boxes) > 0:
                st.success("✅ Objek terdeteksi:")
                for box in results[0].boxes:
                    cls = int(box.cls)
                    label = model.names[cls]
                    conf = float(box.conf)
                    st.write(f"- {label} ({conf:.2f})")
            else:
                st.warning("⚠️ Tidak ada objek terdeteksi.")

    # ==========================
    # ResNet50 - Klasifikasi RPS
    # ==========================
    else:
        model = load_resnet_model()
        st.subheader("✋ Jalankan Klasifikasi RPS")

        if st.button("Jalankan Klasifikasi"):
            # Preprocessing gambar
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            # Prediksi
            preds = model.predict(img_array)
            class_names = ["Rock", "Paper", "Scissors"]
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(tf.nn.softmax(preds))

            st.success(f"🧩 Prediksi: **{predicted_class}** ({confidence*100:.2f}%)")

# ==========================
# Info Model
# ==========================
with st.sidebar.expander("ℹ️ Info Model"):
    if model_choice.startswith("PyTorch"):
        st.write("""
        **Model:** YOLOv8  
        **Fungsi:** Deteksi apakah seseorang memakai masker atau tidak.  
        **Output:** Label seperti `mask` dan `no_mask`.
        """)
    else:
        st.write("""
        **Model:** ResNet50  
        **Fungsi:** Klasifikasi gambar ke dalam tiga kelas — Rock, Paper, Scissors.  
        **Output:** `Rock`, `Paper`, `Scissors`.
        """)
