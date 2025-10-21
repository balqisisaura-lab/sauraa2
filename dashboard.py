import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# ==============================
# 🔧 Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# ==============================
# 🔘 Sidebar Pilih Model
# ==============================
selected_model = st.sidebar.radio(
    "Pilih Model:",
    ["YOLO Object Detection", "TensorFlow Rock–Paper–Scissors"]
)

# ==============================
# 🎯 BAGIAN 1: YOLO
# ==============================
if selected_model == "YOLO Object Detection":
    st.header("🧠 Model PyTorch - YOLO")

    @st.cache_resource
    def load_yolo_model():
        return YOLO("model/Balqis_Isaura_Laporan 4.pt")

    try:
        yolo_model = load_yolo_model()
        st.success("✅ Model YOLO berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        st.stop()

    # Upload gambar
    uploaded_file = st.file_uploader("📸 Upload gambar untuk deteksi", type=["jpg", "jpeg", "png"], key="yolo")
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(image, use_container_width=True)

        if st.button("🔍 Jalankan Deteksi YOLO"):
            with st.spinner("Mendeteksi objek..."):
                results = yolo_model.predict(image, conf=0.2, imgsz=640)  # turunkan conf agar lebih sensitif
                annotated_img = results[0].plot()

                with col2:
                    st.subheader("Hasil Deteksi")
                    st.image(annotated_img, use_container_width=True)

                boxes = results[0].boxes
                st.markdown("### 📋 Detail Deteksi")
                if len(boxes) > 0:
                    for i, box in enumerate(boxes, start=1):
                        cls_name = yolo_model.names[int(box.cls)]
                        conf = float(box.conf[0])
                        st.write(f"**{i}. {cls_name}** — Confidence: `{conf:.2%}`")
                else:
                    st.warning("⚠️ Tidak ada objek terdeteksi.")

# ==============================
# 🧠 BAGIAN 2: TENSORFLOW
# ==============================
elif selected_model == "TensorFlow Rock–Paper–Scissors":
    st.header("🧠 Model TensorFlow - Rock Paper Scissors")

    @st.cache_resource
    def load_tf_model():
        model = tf.keras.models.load_model("model/model_fixed.h5", compile=False)
        return model

    try:
        tf_model = load_tf_model()
        st.success("✅ Model TensorFlow berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Model tidak bisa dimuat: {e}")
        st.info("""
        💡 **Tips Perbaikan:**
        1. Pastikan file `model_fixed.h5` ada di folder `model/`.
        2. Pastikan model dilatih dengan 3 kelas: Rock, Paper, Scissors.
        3. Kalau masih error, coba convert ulang:
        ```python
        import tensorflow as tf
        model = tf.keras.models.load_model('model_fixed.h5', compile=False)
        model.save('model_fixed_new.keras')
        ```
        """)
        st.stop()

    uploaded_file = st.file_uploader("📸 Upload gambar untuk prediksi", type=["jpg", "jpeg", "png"], key="tf")

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar Input")
            st.image(image, use_container_width=True)

        if st.button("🔮 Jalankan Prediksi TensorFlow"):
            with st.spinner("Melakukan prediksi..."):
                try:
                    # Preprocessing
                    img = image.resize((224, 224))
                    img_array = np.array(img)

                    # Convert RGBA → RGB
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)

                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                    # Prediksi
                    preds = tf_model.predict(img_array, verbose=0)
                    predicted_class = np.argmax(preds[0])
                    confidence = float(np.max(preds[0]))

                    class_names = ["Rock", "Paper", "Scissors"]

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", class_names[predicted_class])
                        st.metric("Confidence", f"{confidence:.2%}")

                        st.markdown("### 📊 Probabilitas Tiap Kelas")
                        for i, prob in enumerate(preds[0]):
                            st.progress(float(prob), text=f"{class_names[i]}: {prob:.4f}")

                except Exception as e:
                    st.error(f"❌ Error saat prediksi: {e}")

st.markdown("---")
st.markdown("**📌 Dibuat oleh Balqis Isaura | Powered by Streamlit 🚀**")
