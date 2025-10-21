import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os

# ===================== CONFIGURASI DASHBOARD =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["YOLO Object Detection", "TensorFlow Rock–Paper–Scissors"]
)

# ==================== MODEL YOLO ====================
if model_choice == "YOLO Object Detection":
    st.header("🎯 Model YOLO - Object Detection")

    try:
        @st.cache_resource
        def load_yolo():
            return YOLO("model/Balqis Isaura_Laporan 4.pt")

        with st.spinner("🔄 Memuat model YOLO..."):
            yolo_model = load_yolo()
        st.success("✅ Model YOLO berhasil dimuat!")

        uploaded_file = st.file_uploader("📸 Upload gambar untuk deteksi:", type=["jpg", "jpeg", "png"], key="yolo")

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🚀 Jalankan Deteksi"):
                with st.spinner("Mendeteksi objek..."):
                    results = yolo_model(image)
                    result_img = results[0].plot()
                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img, use_column_width=True)

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
        st.error(f"❌ Error: {e}")
        st.info("Pastikan file `.pt` ada di folder `model/`.")

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow Rock–Paper–Scissors":
    st.header("🧠 Model TensorFlow - Rock Paper Scissors")

    try:
        FILE_ID = "1uYmpPANnUKNKBaRHCOlylWV7t3fDgPp2"
        MODEL_PATH = "model_resnet50_fixed_new.keras"

        # Unduh model dari Google Drive jika belum ada
        if not os.path.exists(MODEL_PATH):
            with st.spinner("⬇️ Mengunduh model dari Google Drive..."):
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            st.success("✅ Model berhasil diunduh!")

        @st.cache_resource
        def load_tf_model():
            return tf.keras.models.load_model(MODEL_PATH, compile=False)

        with st.spinner("🔄 Memuat model TensorFlow..."):
            model = load_tf_model()
        st.success("✅ Model TensorFlow berhasil dimuat!")

        class_names = ["Rock", "Paper", "Scissors"]

        uploaded_file = st.file_uploader("📸 Upload gambar untuk prediksi:", type=["jpg", "jpeg", "png"], key="tf")

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔮 Prediksi Gambar", type="primary"):
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

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", predicted_class)
                        st.metric("Confidence", f"{confidence:.2%}")

                    with st.expander("📊 Probabilitas Tiap Kelas"):
                        for i, prob in enumerate(predictions[0]):
                            st.progress(float(prob), text=f"{class_names[i]}: {prob:.4f}")

    except Exception as e:
        st.error(f"❌ Model tidak bisa dimuat: {str(e)}")
        st.info("""
        💡 **Tips:**
        1. Pastikan file model `.keras` bisa diakses publik dari Google Drive  
        2. Model kamu harus punya 3 output kelas: Rock, Paper, Scissors  
        3. Jika tetap error, coba convert ulang di lokal:
           ```python
           import tensorflow as tf
           model = tf.keras.models.load_model('model_resnet50_fixed_new.keras', compile=False)
           model.save('model_resnet50_repaired.keras')
           ```
        """)

st.markdown("---")
st.markdown("**📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")
