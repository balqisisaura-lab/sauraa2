import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import gdown
import os

# ==========================
# Setup Halaman
# ==========================
st.set_page_config(page_title="Dashboard Deteksi & Klasifikasi", page_icon="🎀", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f6f2ff;
    }
    .title {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 40px;
        color: #2d2d7c;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
    }
    .card {
        background-color: #ffe3f2;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>OBJECTIVES</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🎯 Object Detection & 🧠 Classification</div>", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # YOLO model
    yolo = YOLO("model/object.pt")

    # Download model klasifikasi dari GDrive
    url = "https://drive.google.com/uc?id=1uYmpPANnUKNKBaRHCOlylWV7t3fDgPp2"
    output = "model_resnet50_2.h5"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    clf = tf.keras.models.load_model(output, compile=False)

    return yolo, clf

yolo_model, clf_model = load_models()

# ==========================
# Layout: 2 Kolom
# ==========================
col1, col2 = st.columns(2)

# ======== Kolom 1: DETEKSI OBJEK ========
with col1:
    st.markdown("<div class='card'><h3>🎯 Object Detection</h3>", unsafe_allow_html=True)
    img_det = st.file_uploader("Upload gambar untuk deteksi", type=["jpg","jpeg","png"], key="deteksi")
    if img_det:
        img = Image.open(img_det)
        st.image(img, caption="Gambar Asli", use_container_width=True)
        result = yolo_model(img)
        res_img = result[0].plot()
        st.image(res_img, caption="Hasil Deteksi", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======== Kolom 2: KLASIFIKASI ========
with col2:
    st.markdown("<div class='card'><h3>🧠 Image Classification</h3>", unsafe_allow_html=True)
    img_cls = st.file_uploader("Upload gambar untuk klasifikasi", type=["jpg","jpeg","png"], key="klasifikasi")
    if img_cls:
        img = Image.open(img_cls).resize((224, 224))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)
        pred = clf_model.predict(img_array)
        label = np.argmax(pred, axis=1)[0]
        st.image(img, caption="Gambar Input", use_container_width=True)
        st.success(f"Prediksi kelas: **{label}**")
    st.markdown("</div>", unsafe_allow_html=True)
