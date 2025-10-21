import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="AI Dashboard - Balqis Isaura", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #2e8bff;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1c60b3;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Dashboard - Balqis Isaura")
st.markdown("### Pilih antara deteksi objek (YOLO) atau klasifikasi gambar (ResNet50)")

# ==============================
# Fungsi untuk Memuat Model
# ==============================
@st.cache_resource
def load_models():
    yolo_model = None
    classifier = None

    # Load YOLO
    try:
        if os.path.exists("model/Balqis Isaura_Laporan 4.pt"):
            yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")
            st.sidebar.success("✅ YOLO model loaded")
        else:
            st.sidebar.error("❌ YOLO model not found in /model/")
    except Exception as e:
        st.sidebar.error(f"⚠️ YOLO load error: {e}")

    # Load TensorFlow Model (.keras)
    try:
        if os.path.exists("model/model_resnet50.keras"):
            classifier = tf.keras.models.load_model("model/model_resnet50.keras", compile=False)
            st.sidebar.success("✅ ResNet50 model loaded")
        else:
            st.sidebar.error("❌ ResNet50 model not found in /model/")
    except Exception as e:
        st.sidebar.error(f"⚠️ TensorFlow load error: {e}")

    return yolo_model, classifier

# ==============================
# Fungsi Klasifikasi TensorFlow
# ==============================
def classify_image_tf(model, img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized).astype("float32") / 255.0

    # Pastikan input berbentuk (1, 224, 224, 3)
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    elif img_array.ndim == 4 and img_array.shape[0] != 1:
        img_array = img_array[:1]

    st.write("🧩 Shape input ke model:", img_array.shape)  # debug info

    # Prediksi
    predictions = model.predict(img_array)
    st.write("📊 Shape output model:", predictions.shape)

    class_labels = ['Rock', 'Paper', 'Scissors']
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_labels[class_index], confidence

# ==============================
# Fungsi Deteksi YOLO
# ==============================
def detect_yolo(model, img):
    results = model(img)
    return results

# ==============================
# Sidebar Navigasi
# ==============================
st.sidebar.header("⚙️ Model Options")
yolo_model, classifier = load_models()
model_choice = st.sidebar.radio(
    "Pilih model yang ingin digunakan:",
    ["YOLO - Mask Detection", "TensorFlow - Rock Paper Scissors"]
)

st.sidebar.markdown("---")
st.sidebar.info("📎 Pastikan file model tersimpan di folder `model/`")

# ==============================
# Upload Gambar
# ==============================
uploaded_file = st.file_uploader(
    "📤 Upload gambar (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Proses Deteksi / Klasifikasi
# ==============================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    if model_choice == "YOLO - Mask Detection":
        st.subheader("🧠 Hasil Deteksi YOLO")

        if yolo_model is not None:
            with st.spinner("Sedang mendeteksi objek..."):
                results = detect_yolo(yolo_model, img)
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

                detections = results[0].boxes
                if len(detections) > 0:
                    st.success(f"✅ {len(detections)} objek terdeteksi")
                else:
                    st.info("ℹ️ Tidak ada objek terdeteksi.")
        else:
            st.error("❌ Model YOLO belum dimuat atau bermasalah.")

    elif model_choice == "TensorFlow - Rock Paper Scissors":
        st.subheader("✋ Hasil Klasifikasi ResNet50")

        if classifier is not None:
            with st.spinner("Sedang mengklasifikasi gambar..."):
                label, confidence = classify_image_tf(classifier, img)
                if label:
                    st.success(f"**Prediksi:** {label}")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    st.progress(confidence)
                else:
                    st.error("⚠️ Terjadi kesalahan saat prediksi.")
        else:
            st.error("❌ Model TensorFlow belum dimuat atau bermasalah.")
else:
    st.info("📸 Silakan upload gambar terlebih dahulu.")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("<center>✨ Made with ❤️ by <b>Balqis Isaura</b> ✨</center>", unsafe_allow_html=True)
