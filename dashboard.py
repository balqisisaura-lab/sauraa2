import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Dashboard Model - Balqis Isaura", layout="wide")

# ==============================
# Fungsi untuk memuat model
# ==============================
@st.cache_resource
def load_models():
    yolo_model = None
    classifier = None
    
    # Load YOLO model (.pt)
    try:
        if os.path.exists("model/Balqis Isaura_Laporan 4.pt"):
            yolo_model = YOLO("model/Balqis Isaura_Laporan 4.pt")
            st.success("✅ Model YOLO berhasil dimuat")
        else:
            st.error("❌ File YOLO tidak ditemukan di folder model/")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
    
    # Load TensorFlow model (.keras)
    try:
        if os.path.exists("model/model_resnet50.keras"):
            classifier = tf.keras.models.load_model("model/model_resnet50.keras")
            st.success("✅ Model TensorFlow berhasil dimuat")
        else:
            st.error("❌ File model_resnet50.keras tidak ditemukan di folder model/")
    except Exception as e:
        st.error(f"❌ Gagal memuat model TensorFlow: {e}")
    
    return yolo_model, classifier

# ==============================
# Fungsi Klasifikasi TensorFlow
# ==============================
def classify_image_tf(model, img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
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
# Dashboard Header
# ==============================
st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("### 📤 Upload gambar untuk deteksi / klasifikasi:")

# ==============================
# Debug Info (Optional)
# ==============================
with st.expander("🔍 Debug Info - Cek File"):
    st.write("Current Directory:", os.getcwd())
    if os.path.exists("model"):
        st.write("Files in model folder:", os.listdir("model"))
    else:
        st.error("Folder 'model' tidak ditemukan!")

# ==============================
# Upload File
# ==============================
uploaded_file = st.file_uploader(
    "Upload file gambar",
    type=["jpg", "jpeg", "png"],
    help="Limit 200MB per file • JPG, JPEG, PNG"
)

# ==============================
# Load Model
# ==============================
yolo_model, classifier = load_models()

# ==============================
# Pilihan Model
# ==============================
st.markdown("---")
st.subheader("Pilih Model:")
model_choice = st.radio(
    "Pilih Model:",
    ["YOLO - Mask Detection", "TensorFlow - Rock Paper Scissors"],
    horizontal=True
)

# ==============================
# Proses Gambar
# ==============================
if uploaded_file:
    st.markdown("---")
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
    
    with col2:
        if model_choice == "YOLO - Mask Detection":
            st.subheader("🧠 Hasil Deteksi YOLO")
            
            if yolo_model is not None:
                with st.spinner("Memproses deteksi..."):
                    results = detect_yolo(yolo_model, img)
                    result_img = results[0].plot()
                    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
                    
                    # Tampilkan informasi deteksi
                    detections = results[0].boxes
                    if len(detections) > 0:
                        st.success(f"✅ Terdeteksi {len(detections)} objek")
                    else:
                        st.info("ℹ️ Tidak ada objek yang terdeteksi")
            else:
                st.error("❌ Model YOLO tidak tersedia. Pastikan file model ada di folder model/")
        
        elif model_choice == "TensorFlow - Rock Paper Scissors":
            st.subheader("✋ Hasil Klasifikasi ResNet50")
            
            if classifier is not None:
                with st.spinner("Memproses klasifikasi..."):
                    label, confidence = classify_image_tf(classifier, img)
                    
                    # Tampilkan hasil
                    st.success(f"**Prediksi:** {label}")
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Progress bar untuk confidence
                    st.progress(confidence)
            else:
                st.error("❌ Model TensorFlow tidak tersedia. Pastikan file model ada di folder model/")

else:
    st.info("📷 Silakan upload gambar terlebih dahulu untuk mulai deteksi atau klasifikasi.")
    
    # Contoh gambar
    st.markdown("---")
    st.markdown("### 📋 Panduan Penggunaan:")
    st.markdown("""
    1. Upload gambar menggunakan tombol di atas
    2. Pilih model yang ingin digunakan:
       - **YOLO**: Untuk deteksi masker wajah
       - **TensorFlow**: Untuk klasifikasi Rock-Paper-Scissors
    3. Lihat hasil prediksi di sebelah kanan
    """)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("Made with ❤️ by Balqis Isaura")
