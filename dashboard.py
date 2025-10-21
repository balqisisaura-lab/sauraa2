import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os

# ===================== KONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

# ===================== STYLE KHUSUS (NEON CYBER UI) =====================
st.markdown("""
    <style>
    body, .stApp {
        background: radial-gradient(circle at top left, #1a0033, #000000);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3 {
        color: white;
        text-align: center;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #330066, #000000);
        color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #8a2be2, #4b0082);
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff00ff, #8a2be2);
        box-shadow: 0px 0px 20px 4px rgba(255, 0, 255, 0.4);
        transform: scale(1.05);
    }
    .card {
        background: linear-gradient(145deg, #220044, #0d001a);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 25px;
        margin: 15px;
        text-align: center;
        box-shadow: 0 0 25px rgba(138,43,226,0.3);
        transition: transform 0.3s ease;
    }
    .card:hover { transform: translateY(-5px); }
    .stProgress > div > div {
        background: linear-gradient(90deg, #8a2be2, #ff00ff);
    }
    div[data-testid="stMetricValue"] {
        color: #ff66ff;
        font-size: 1.5rem;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===================== JUDUL =====================
st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# ===================== DUA KOLOM =====================
col1, col2 = st.columns(2)

# ===================== KOLOM 1 - YOLO =====================
with col1:
    st.markdown('<div class="card"><h2>📸 YOLO Object Detection</h2>', unsafe_allow_html=True)
    try:
        @st.cache_resource
        def load_yolo():
            return YOLO("model/Balqis Isaura_Laporan 4.pt")

        with st.spinner("🔄 Memuat model YOLO..."):
            yolo_model = load_yolo()
        st.success("✅ Model YOLO berhasil dimuat!")

        uploaded_file_yolo = st.file_uploader("Upload gambar untuk deteksi:", type=["jpg", "jpeg", "png"], key="yolo")

        if uploaded_file_yolo:
            image = Image.open(uploaded_file_yolo)
            st.image(image, caption="📷 Gambar Input", use_column_width=True)

            if st.button("🚀 Jalankan Deteksi"):
                with st.spinner("Mendeteksi objek..."):
                    results = yolo_model(image)
                    result_img = results[0].plot()
                    st.image(result_img, caption="🎯 Hasil Deteksi", use_column_width=True)

                    st.markdown("### 📋 Detail Deteksi")
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes, 1):
                            label = yolo_model.names[int(box.cls)]
                            conf = box.conf[0]
                            st.write(f"{i}. **{label}** — Confidence: {conf:.2%}")
                    else:
                        st.info("Tidak ada objek terdeteksi.")
    except Exception as e:
        st.error(f"❌ Error YOLO: {e}")
        st.info("Pastikan file `.pt` ada di folder `model/`.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== KOLOM 2 - TENSORFLOW =====================
with col2:
    st.markdown('<div class="card"><h2>🧠 TensorFlow Rock–Paper–Scissors</h2>', unsafe_allow_html=True)
    try:
        FILE_ID = "1uYmpPANnUKNKBaRHCOlylWV7t3fDgPp2"  # ID dari link Drive kamu
        MODEL_PATH = "model_resnet50_balqis.h5"

        if not os.path.exists(MODEL_PATH):
            with st.spinner("⬇️ Mengunduh model (.h5) dari Google Drive..."):
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("✅ Model berhasil diunduh dari Google Drive!")

        @st.cache_resource
        def load_tf_model():
            return tf.keras.models.load_model(MODEL_PATH, compile=False)

        with st.spinner("🔄 Memuat model TensorFlow (.h5)..."):
            model = load_tf_model()
        st.success("✅ Model TensorFlow berhasil dimuat!")

        class_names = ["Rock", "Paper", "Scissors"]

        uploaded_file_tf = st.file_uploader("Upload gambar untuk klasifikasi:", type=["jpg", "jpeg", "png"], key="tf")

        if uploaded_file_tf:
            image = Image.open(uploaded_file_tf)
            st.image(image, caption="📷 Gambar Input", use_column_width=True)

            if st.button("🔮 Prediksi Gambar"):
                with st.spinner("Melakukan prediksi..."):
                    img_array = np.array(image.resize((224, 224))) / 255.0
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array]*3, axis=-1)
                    elif img_array.shape[-1] == 4:
                        img_array = img_array[..., :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    predictions = model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_index]
                    confidence = predictions[0][predicted_index]

                    st.metric("Kelas Prediksi", predicted_class)
                    st.metric("Confidence", f"{confidence:.2%}")

                    with st.expander("📊 Probabilitas Tiap Kelas"):
                        for i, prob in enumerate(predictions[0]):
                            st.progress(float(prob), text=f"{class_names[i]}: {prob:.4f}")
    except Exception as e:
        st.error(f"❌ Model TensorFlow gagal dimuat: {str(e)}")
        st.info("""
        💡 **Tips:**
        1. Pastikan link Drive kamu publik  
        2. Model harus punya 3 output kelas: Rock, Paper, Scissors  
        3. Jika tetap error, coba convert ulang ke format `.keras`:
           ```python
           import tensorflow as tf
           model = tf.keras.models.load_model('model_resnet50_balqis.h5', compile=False)
           model.save('model_resnet50_balqis.keras')
           ```
        """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("**📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")
