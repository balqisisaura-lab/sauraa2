import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==============================
# Fungsi untuk memuat model
# ==============================
@st.cache_resource
def load_models():
    yolo_model = None
    classifier = None

    # Load YOLO model (.pt)
    try:
        yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")  # path disesuaikan
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model YOLO: {e}")
    
    # Load TensorFlow model (.keras)
    try:
        classifier = tf.keras.models.load_model("model/model_resnet50.keras")  # path disesuaikan
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model TensorFlow: {e}")

    return yolo_model, classifier


# ==============================
# Fungsi Klasifikasi TensorFlow
# ==============================
def classify_image_tf(model, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalisasi

    predictions = model.predict(img_array)
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
# Dashboard Streamlit
# ==============================
st.set_page_config(page_title="Dashboard Model - Balqis Isaura", layout="wide")
st.title("🎯 Dashboard Model - Balqis Isaura")

st.markdown("### 📤 Upload gambar untuk deteksi / klasifikasi:")
uploaded_file = st.file_uploader("Upload file gambar", type=["jpg", "jpeg", "png"])

# Load model
yolo_model, classifier = load_models()

# Pilihan model
model_choice = st.radio(
    "Pilih Model:",
    ["YOLO - Mask Detection", "TensorFlow - Rock Paper Scissors"]
)

# ==============================
# Jika gambar diupload
# ==============================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    if model_choice == "YOLO - Mask Detection":
        st.subheader("🧠 Deteksi Menggunakan YOLO")
        if yolo_model is not None:
            results = detect_yolo(yolo_model, img)
            result_img = results[0].plot()  # hasil bounding box
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
        else:
            st.error("❌ Model YOLO belum dimuat atau bermasalah.")

    elif model_choice == "TensorFlow - Rock Paper Scissors":
        st.subheader("✋ Klasifikasi Gunting-Batu-Kertas (ResNet50)")
        if classifier is not None:
            label, confidence = classify_image_tf(classifier, img)
            st.success(f"**Hasil:** {label} ({confidence*100:.2f}%)")
        else:
            st.error("❌ Model TensorFlow belum dimuat atau bermasalah.")

else:
    st.info("📷 Silakan upload gambar terlebih dahulu untuk mulai deteksi atau klasifikasi.")
