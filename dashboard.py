import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os 
from io import BytesIO

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="AI Vision - Mask Detection & Hand Gesture",
    page_icon="🤖",
    layout="wide"
)

# ========================== CUSTOM CSS ==========================
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.1);
        }
        h1, h2, h3, h4 {
            color: #00d4ff !important;
        }
        .stButton>button {
            background-color: #00d4ff;
            color: black;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #00a2cc;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.title("🤖 AI Vision - Mask Detection & Hand Gesture Recognition")
st.markdown("### Smart Vision System using YOLO and TensorFlow")
st.markdown("---")

# ========================== LOAD MODELS ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/AI_Vision_Mask.pt")  # ganti path model YOLO kamu
    gesture_model = tf.keras.models.load_model("model/gesture_model.h5")  # ganti path model gesture kamu
    return yolo_model, gesture_model

yolo_model, gesture_model = load_models()

# ========================== UPLOAD IMAGE ==========================
uploaded_file = st.file_uploader("📤 Upload Gambar untuk Deteksi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    # ========================== YOLO MASK DETECTION ==========================
    st.subheader("😷 Deteksi Masker (YOLO)")
    with st.spinner("Model sedang mendeteksi..."):
        results = yolo_model.predict(image)
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Hasil Deteksi Masker", use_container_width=True)

    st.markdown("---")

    # ========================== HAND GESTURE CLASSIFICATION ==========================
    st.subheader("✋ Klasifikasi Gesture Tangan")
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("Model sedang mengenali gesture..."):
        prediction = gesture_model.predict(img_array)
        gesture_classes = ['Rock', 'Paper', 'Scissors']  # sesuaikan dengan kelas modelmu
        predicted_class = np.argmax(prediction)
        final_result = gesture_classes[predicted_class]

    st.success(f"Gesture terdeteksi: **{final_result}**")

    # ========================== EFEK VISUAL ==========================
    if final_result == 'Rock':
        st.balloons()
    elif final_result == 'Paper':
        pass  # efek salju dihapus
    else:
        st.balloons()

# ========================== SIDEBAR ==========================
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.markdown("""
Gunakan panel ini untuk melihat informasi sistem dan pengaturan lainnya.
""")
st.sidebar.markdown("---")
st.sidebar.info("""
**AI Vision** menggunakan kombinasi model:
- YOLO untuk deteksi masker  
- CNN (TensorFlow) untuk klasifikasi gesture
""")

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: #00d4ff;'>
        © 2025 <b>AI Vision</b> | Developed by Balqis Isaura 💙
    </p>
    """,
    unsafe_allow_html=True
)
