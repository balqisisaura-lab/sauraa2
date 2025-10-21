import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

# ==========================
# Fungsi untuk memuat model
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model_yolo.pt")  # Ganti dengan nama file YOLO kamu
    except Exception as e:
        yolo_model = None
        st.warning(f"⚠️ Gagal memuat model YOLO: {e}")

    try:
        classifier = tf.keras.models.load_model("model_resnet50.keras")  # model transfer learning kamu
    except Exception as e:
        classifier = None
        st.warning(f"⚠️ Gagal memuat model TensorFlow: {e}")

    return yolo_model, classifier


# ==========================
# Inisialisasi model
# ==========================
st.set_page_config(page_title="Dashboard Model - Balqis Isaura", layout="wide")
st.title("🎯 Dashboard Model - Balqis Isaura")
st.caption("📸 Upload gambar untuk deteksi / klasifikasi")

yolo_model, classifier = load_models()


# ==========================
# Upload file
# ==========================
uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Pilihan mode
    mode = st.radio(
        "Pilih Mode:",
        ["Deteksi Objek (YOLO)", "Klasifikasi (ResNet50)"],
        horizontal=True
    )

    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK (YOLO)
    # ==========================
    if mode == "Deteksi Objek (YOLO)":
        if yolo_model:
            if st.button("🚀 Jalankan Deteksi YOLO"):
                try:
                    results = yolo_model.predict(np.array(img))
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

                    # Tampilkan detail deteksi
                    if len(results[0].boxes) > 0:
                        st.subheader("📋 Detail Deteksi")
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            st.write(f"🟢 Kelas: {yolo_model.names[cls]} | 🔢 Confidence: {conf:.2f}")
                    else:
                        st.warning("⚠️ Tidak ada objek terdeteksi.")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat deteksi: {e}")
        else:
            st.warning("⚠️ Model YOLO belum dimuat.")

    # ==========================
    # MODE KLASIFIKASI (TRANSFER LEARNING)
    # ==========================
    elif mode == "Klasifikasi (ResNet50)":
        if classifier:
            if st.button("🧠 Jalankan Klasifikasi"):
                try:
                    # Preprocessing gambar (sesuai model transfer learning)
                    img_resized = img.resize((224, 224))
                    x = image.img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)   # (1, 224, 224, 3)
                    x = x / 255.0

                    # Prediksi
                    pred = classifier.predict(x)
                    kelas = np.argmax(pred, axis=1)[0]

                    st.success(f"✅ Hasil klasifikasi: **Kelas {kelas}**")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
        else:
            st.warning("⚠️ Model TensorFlow belum dimuat.")
