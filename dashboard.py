import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import cv2
import io

st.set_page_config(page_title="Dashboard Model - Balqis Isaura", layout="wide")

st.title("🎯 Dashboard Model - Balqis Isaura")

st.write("📤 Upload gambar untuk deteksi / klasifikasi:")

# ============================================
# BAGIAN UPLOAD GAMBAR
# ============================================
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ============================================
# PILIH MODEL
# ============================================
st.sidebar.title("Pilih Model:")
model_type = st.sidebar.radio("",
    ["PyTorch - YOLO", "TensorFlow - RockPaperScissors"],
    label_visibility="collapsed"
)

# ============================================
# FUNGSI LOAD MODEL
# ============================================
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("model/Balqis Isaura_Laporan 4.pt")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        return None


@st.cache_resource
def load_resnet_model():
    try:
        model = tf.keras.models.load_model("model_resnet50.keras")

        # pastikan model punya input layer
        if not hasattr(model, "input_shape"):
            model.build((None, 224, 224, 3))
        return model

    except Exception as e:
        try:
            # 🔧 fallback: buat ulang model dengan input layer
            base_model = tf.keras.models.load_model("model_resnet50.keras", compile=False)
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.layers.Resizing(224, 224)(inputs)
            x = tf.keras.layers.Rescaling(1./255)(x)
            x = base_model(x, training=False)
            outputs = tf.keras.layers.Softmax()(x)
            fixed_model = tf.keras.Model(inputs, outputs)
            return fixed_model
        except Exception as e2:
            st.error(f"❌ Gagal memuat model TensorFlow: {e2}")
            return None

# ============================================
# FUNGSI YOLO DETEKSI
# ============================================
def run_yolo_detection(image):
    model = load_yolo_model()
    if model is None:
        return None

    results = model(image)
    result_img = results[0].plot()

    st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

    if len(results[0].boxes) == 0:
        st.warning("⚠️ Tidak ada objek terdeteksi.")
    else:
        st.success("✅ Objek berhasil terdeteksi!")
        for box in results[0].boxes:
            st.write({
                "Kelas": int(box.cls[0]),
                "Confidence": float(box.conf[0])
            })

# ============================================
# FUNGSI KLASIFIKASI TENSORFLOW
# ============================================
def run_rps_classification(image):
    model = load_resnet_model()
    if model is None:
        return None

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    class_names = ["Rock", "Paper", "Scissors"]
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.image(image, caption=f"Gambar Input", use_container_width=True)
    st.markdown(f"### 🧠 Prediksi: **{predicted_class}** ({confidence:.2f} confidence)")

# ============================================
# BAGIAN UTAMA
# ============================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    if model_type == "PyTorch - YOLO":
        st.subheader("📦 Model Deteksi Objek YOLO")
        run_yolo_detection(image)

    elif model_type == "TensorFlow - RockPaperScissors":
        st.subheader("🧠 Model Klasifikasi Rock-Paper-Scissors")
        run_rps_classification(image)
else:
    st.info("📂 Upload gambar terlebih dahulu untuk mulai deteksi atau klasifikasi.")

