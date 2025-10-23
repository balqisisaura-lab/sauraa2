import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================

@st.cache_resource
def load_models():
    try:
        # Load YOLO model
        yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")

        # Load Classification model (your own .h5 file)
        classifier = tf.keras.models.load_model("model/compressed.h5")

        return yolo_model, classifier

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None


# ==========================
# Preprocess Function
# ==========================

def preprocess_image(img):
    img = img.resize((224, 224))  # sesuaikan ukuran input model kamu
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisasi (sesuaikan jika model kamu beda)
    return img_array


# ==========================
# Streamlit UI
# ==========================

st.set_page_config(page_title="Pizza Classifier", page_icon="🍕", layout="wide")

st.title("🍕 Pizza Classification & Object Detection App")
st.write("Upload gambar untuk mendeteksi dan mengklasifikasi menggunakan model kamu sendiri.")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

# Load Models
yolo_model, classification_model = load_models()

if uploaded_file is not None and yolo_model is not None and classification_model is not None:
    # Tampilkan gambar yang diupload
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # ==========================
    # YOLO Object Detection
    # ==========================
    st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
    results = yolo_model(img)

    # Tampilkan hasil deteksi (bounding box)
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Hasil Deteksi YOLO", use_container_width=True)

    # ==========================
    # Classification Model
    # ==========================
    st.subheader("🤖 Hasil Klasifikasi (Compressed Model)")
    preprocessed_img = preprocess_image(img)
    predictions = classification_model.predict(preprocessed_img)

    # Ambil kelas dengan probabilitas tertinggi
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Ubah sesuai label model kamu
    # Misalnya: 0 = Bukan Pizza, 1 = Pizza
    if predicted_class == 1:
        st.success(f"✅ Prediksi: **Pizza** ({confidence*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"❌ Prediksi: **Bukan Pizza** ({confidence*100:.2f}%)")
        st.snow()

else:
    st.info("📸 Silakan unggah gambar terlebih dahulu dan pastikan model sudah tersedia di folder 'model/'.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Dibuat oleh Balqis Isaura 💛 | Streamlit + YOLO + TensorFlow")
