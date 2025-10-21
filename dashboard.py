import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
    layout="centered"
)

# ==========================
# Custom CSS for Tech Theme
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        .main {
            background-color: #161a23;
            border-radius: 15px;
            padding: 30px;
        }
        h1, h2, h3 {
            color: #61dafb;
            text-align: center;
        }
        .stButton>button {
            background-color: #61dafb;
            color: black;
            border-radius: 10px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #21a1f1;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_resnet50.keras", compile=False)
    return model

model = load_model()

# ==========================
# Kelas target
# ==========================
class_labels = ["Kelas 1", "Kelas 2", "Kelas 3"]  # ubah sesuai label aslimu

# ==========================
# UI Header
# ==========================
st.title("🤖 Image Classifier using ResNet50")
st.markdown("Upload gambar dan biarkan AI menebak kelasnya 🚀")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# ==========================
# Prediksi
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Prediksi
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")
    st.markdown(f"**Kelas Prediksi:** {class_labels[pred_class]}")
    st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%}")
