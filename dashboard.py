import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==============================
# CONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="AI Dashboard - Balqis Isaura",
    page_icon="🤖",
    layout="centered"
)

# ==============================
# TEMA TEKNOLOGI (DARK MODE)
# ==============================
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: #e0e0e0;
        }
        .main {
            background-color: #161a23;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 0 15px rgba(0,0,0,0.4);
        }
        h1, h2, h3 {
            color: #61dafb;
            text-align: center;
        }
        .stButton>button {
            background-color: #61dafb;
            color: black;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #21a1f1;
            color: white;
        }
        .stRadio>div>label {
            color: #61dafb !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    yolo_model = None
    classifier = None
    # Load YOLO model (.pt)
    try:
        yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")  # path sesuai file kamu
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model YOLO: {e}")

    # Load TensorFlow model (.keras)
    try:
        classifier = tf.keras.models.load_model("model/model_resnet50.keras", compile=False)
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model TensorFlow: {e}")

    return yolo_model, classifier


# ==============================
# FUNGSI KLASIFIKASI
# ==============================
def classify_image_tf(model, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    class_labels = ['Rock', 'Paper', 'Scissors']  # ubah jika label berbeda
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_labels[class_index], confidence


# ==============================
# FUNGSI DETEKSI YOLO
# ==============================
def detect_yolo(model, img):
    results = model(img)
    return results


# ==============================
# DASHBOARD
# ==============================
st.title("🤖 AI Detection & Classification Dashboard")
st.markdown("### Pilih model dan upload gambar untuk melakukan prediksi 🎯")

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload file gambar", type=["jpg", "jpeg", "png"])

# Load model
yolo_model, classifier = load_models()

# Pilihan model
model_choice = st.radio(
    "🔍 Pilih jenis model:",
    ["YOLO - Object Detection", "ResNet50 - Image Classification"]
)

# ==============================
# JIKA ADA GAMBAR YANG DIUPLOAD
# ==============================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diupload", use_container_width=True)
    st.markdown("---")

    if model_choice == "YOLO - Object Detection":
        st.subheader("🧠 Deteksi Menggunakan YOLO")
        if yolo_model is not None:
            with st.spinner("Sedang mendeteksi objek..."):
                results = detect_yolo(yolo_model, img)
                result_img = results[0].plot()  # hasil bounding box
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)
        else:
            st.error("❌ Model YOLO belum dimuat atau terjadi kesalahan saat loading.")

    elif model_choice == "ResNet50 - Image Classification":
        st.subheader("✋ Klasifikasi Gambar (Rock, Paper, Scissors)")
        if classifier is not None:
            with st.spinner("Sedang melakukan klasifikasi..."):
                label, confidence = classify_image_tf(classifier, img)
                st.success(f"**Hasil Prediksi:** {label}")
                st.info(f"📊 Tingkat Keyakinan: {confidence*100:.2f}%")
        else:
            st.error("❌ Model TensorFlow belum dimuat atau terjadi kesalahan saat loading.")

else:
    st.info("📁 Silakan upload gambar terlebih dahulu untuk memulai prediksi.")
