import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ====================================
# LOAD MODEL DENGAN FIX UNTUK PYTORCH >= 2.6
# ====================================
@st.cache_resource
def load_model():
    # Izinkan class DetectionModel untuk unpickling
    torch.serialization.add_safe_globals([__import__('ultralytics').nn.tasks.DetectionModel])
    
    # Load model YOLO kamu
    model_path = "model/Balqis Isaura_Laporan 4.pt"
    model = YOLO(model_path)
    return model


# ====================================
# DASHBOARD TITLE
# ====================================
st.title("🎯 Deteksi Objek dengan YOLO — Balqis Isaura")
st.markdown("Upload gambar untuk mendeteksi objek menggunakan model kamu.")

# ====================================
# LOAD MODEL
# ====================================
model = load_model()

# ====================================
# UPLOAD GAMBAR
# ====================================
uploaded_file = st.file_uploader("📤 Upload gambar di sini:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")
    
    # Tampilkan gambar asli
    st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    # Tombol prediksi
    if st.button("🚀 Jalankan Deteksi"):
        with st.spinner("Model sedang mendeteksi..."):
            # Jalankan prediksi
            results = model.predict(source=np.array(image), conf=0.5, verbose=False)
            
            # Ambil hasil gambar
            result_image = results[0].plot()  # gambar hasil deteksi
            
            # Tampilkan hasil
            st.image(result_image, caption="Hasil Deteksi", use_container_width=True)
            st.success("✅ Deteksi selesai!")

            # (Opsional) tampilkan label dan confidence
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.subheader("📋 Detil Deteksi:")
                for box in boxes:
                    label = results[0].names[int(box.cls)]
                    conf = float(box.conf)
                    st.write(f"- **{label}** ({conf:.2f})")
            else:
                st.warning("Tidak ada objek terdeteksi pada gambar ini.")
