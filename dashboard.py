import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gdown
import os
import io

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="The Future of Image Analytics - Balqis Isaura",
    page_icon="🔮",
    layout="wide"
)

# ===================== STYLE SESUAI GAMBAR (DARK MODE UNGU) =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

/* Main Background */
.stApp {
    background-color: #000000;
    color: #FFFFFF;
    font-family: 'Poppins', sans-serif;
}

/* Menyembunyikan elemen bawaan Streamlit */
#MainMenu, header, footer {visibility: hidden;}

/* --- Header & Title Styling --- */
.header-title-container {
    padding-top: 50px;
    padding-bottom: 30px;
}

.header-title {
    font-size: 4.5rem; /* Ukuran font besar untuk judul utama */
    font-weight: 800;
    line-height: 1.1;
    color: #E7C9FF; /* Warna teks terang */
    margin: 0;
}

.header-subtitle {
    font-size: 1.2rem;
    font-weight: 400;
    color: #B58CF2; /* Warna teks ungu lebih muda */
    margin-top: -10px;
}

/* Kotak Deskripsi */
.description-box {
    background-color: #2D2D2D; /* Latar belakang abu-abu gelap */
    color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #4B0082; /* Border ungu tua */
    line-height: 1.6;
    font-size: 0.95rem;
}

/* --- Card Analisis --- */
.analysis-card {
    background: linear-gradient(135deg, #4B0082 0%, #29004B 100%); /* Gradien Ungu Gelap */
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 4px 15px rgba(124, 77, 255, 0.4); /* Shadow ungu */
    height: 100%; /* Memastikan tinggi kolom sama */
}

/* Judul Fitur (Deteksi Objek/Klasifikasi Gambar) */
.analysis-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F0E6FF; /* Teks sangat terang */
    margin-bottom: 20px;
}

/* Style untuk area upload file */
.stFileUploader > div {
    background-color: #4B0082; /* Latar belakang area upload */
    color: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    border: 2px dashed #9D8CFF; /* Border ungu terang dashed */
}

/* Style untuk tombol Run/Prediksi */
.stButton>button {
    background-color: #9D8CFF;
    color: #000000;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s;
}

.stButton>button:hover {
    background-color: #7C4DFF;
    color: #FFFFFF;
}

/* Teks 'Hasil Analisis' */
.analysis-result-label {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F0E6FF;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Teks Hasil Deteksi/Klasifikasi */
.result-text {
    font-size: 1.2rem;
    color: #FFFFFF;
    font-weight: 600;
}

/* Gaya untuk angka Metrik (optional) */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    color: #9D8CFF !important;
}
[data-testid="stMetricLabel"] {
    color: #E7C9FF !important;
}

</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL (Dipindah ke atas untuk di-cache) =====================

# Load YOLO Model (Deteksi Masker)
@st.cache_resource
def load_yolo():
    try:
        # Asumsikan file Balqis Isaura_Laporan 4.pt ada di folder model/
        return YOLO("model/Balqis Isaura_Laporan 4.pt")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        return None

# Load TensorFlow Model (Klasifikasi RPS)
@st.cache_resource
def load_tf_model():
    MODEL_PATH = "model_resnet50_balqis.h5"
    FILE_ID = "1uYmpPANnUKNKBaRHCOlylWV7t3fDgPp2" # ID GDrive Anda

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Mengunduh model TensorFlow dari Google Drive..."):
            try:
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model TensorFlow: {e}")
                return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Gagal memuat model TensorFlow: {e}")
        return None

# ===================== TAMPILAN UTAMA (HEADER & LAYOUT) =====================

col_title, col_desc = st.columns([0.4, 0.6], gap="large")

with col_title:
    st.markdown(f"""
    <div class="header-title-container">
        <div class="header-title">The Future of</div>
        <div class="header-title" style="margin-top:-20px;">Image Analytics</div>
        <div class="header-subtitle">by Balqis Isaura</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Area kosong untuk mengimbangi kolom deskripsi di atas
    st.markdown("<br><br><br>", unsafe_allow_html=True) 

with col_desc:
    # Menggunakan kotak deskripsi sesuai desain visual
    st.markdown("""
    <div class="description-box">
    Ukur kepatuhan dan kenali pola dalam gambar dengan teknologi AI. Dashboard ini mengkhususkan diri dalam **Deteksi Objek** untuk memantau penggunaan masker, serta **Klasifikasi Gambar** untuk mengidentifikasi pola visual (**Batu, Gunting, Kertas**) dari gambar diam yang diunggah.
    </div>
    """, unsafe_allow_html=True)

# Garis pemisah atau area kosong
st.markdown("---") 

# ===================== DUA KOLOM FITUR (DETEKSI & KLASIFIKASI) =====================

col_deteksi, col_klasifikasi = st.columns(2, gap="large")

# ----------------- KOLOM DETEKSI OBJEK (Masker) -----------------
with col_deteksi:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown('<div class="analysis-title">Deteksi Objek</div>', unsafe_allow_html=True)
    
    yolo_model = load_yolo()

    if yolo_model:
        uploaded_file_yolo = st.file_uploader(
            "Upload Gambar (JPG, JPEG, PNG)", 
            type=["jpg", "jpeg", "png"], 
            key="yolo",
            help="Limit 200MB per file"
        )
        
        st.markdown('<div class="analysis-result-label">Hasil analisis :</div>', unsafe_allow_html=True)
        
        if uploaded_file_yolo:
            image = Image.open(uploaded_file_yolo)
            
            # Tampilkan gambar yang diunggah
            st.image(image, caption="Gambar Input", use_container_width=True)

            if st.button("🚀 Jalankan Deteksi Masker", key="btn_yolo", use_container_width=True):
                with st.spinner("🔍 Mendeteksi penggunaan masker..."):
                    try:
                        # Jalankan inferensi YOLO
                        results = yolo_model(image)
                        
                        # Tampilkan hasil deteksi
                        result_img = results[0].plot(
                            line_width=2, 
                            font_size=16, 
                            labels=True, 
                            conf=True
                        )
                        st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)

                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            st.markdown("---")
                            st.markdown("#### Detail Pelanggaran:")
                            
                            # Hitung statistik
                            stats = {"Memakai Masker": 0, "Tidak Memakai Masker": 0}
                            
                            for i, box in enumerate(boxes, 1):
                                label = yolo_model.names[int(box.cls)]
                                conf = box.conf[0]
                                stats[label] += 1
                                st.write(f"**{i}.** {label} — Confidence: **{conf:.2%}**")
                            
                            st.markdown("---")
                            st.metric("Total Wajah Terdeteksi", len(boxes))
                            st.metric("Total TIDAK Memakai Masker", stats.get("Tidak Memakai Masker", 0))

                        else:
                            st.info("ℹ️ Tidak ada wajah terdeteksi.")
                    except Exception as e:
                        st.error(f"❌ Error saat menjalankan deteksi: {e}")
        else:
            st.info("⬆️ Silakan unggah gambar untuk memulai deteksi.")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- KOLOM KLASIFIKASI GAMBAR (RPS) -----------------
with col_klasifikasi:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown('<div class="analysis-title">Klasifikasi Gambar</div>', unsafe_allow_html=True)
    
    tf_model = load_tf_model()
    class_names = ["Rock", "Paper", "Scissors"]

    if tf_model:
        uploaded_file_tf = st.file_uploader(
            "Upload Gambar (JPG, JPEG, PNG)", 
            type=["jpg", "jpeg", "png"], 
            key="tf",
            help="Limit 200MB per file"
        )
        
        st.markdown('<div class="analysis-result-label">Hasil analisis :</div>', unsafe_allow_html=True)

        if uploaded_file_tf:
            image = Image.open(uploaded_file_tf)
            st.image(image, caption="Gambar Input", use_container_width=True)

            if st.button("🔮 Prediksi RPS", key="btn_tf", use_container_width=True):
                with st.spinner("🤖 Melakukan prediksi..."):
                    try:
                        # Preprocessing gambar
                        img_array = np.array(image.resize((224, 224))) / 255.0
                        if len(img_array.shape) == 2:
                             # Convert grayscale to RGB if needed
                            img_array = np.stack([img_array]*3, axis=-1) 
                        elif img_array.shape[-1] == 4:
                            # Remove alpha channel if present
                            img_array = img_array[..., :3] 
                            
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Prediksi
                        predictions = tf_model.predict(img_array, verbose=0)
                        predicted_index = np.argmax(predictions[0])
                        predicted_class = class_names[predicted_index]
                        confidence = predictions[0][predicted_index]

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("🎯 Kelas Prediksi", predicted_class)
                        with col_b:
                            st.metric("📊 Confidence", f"{confidence:.2%}")

                        st.markdown("---")
                        with st.expander("✨ Lihat Probabilitas Tiap Kelas"):
                            for i, prob in enumerate(predictions[0]):
                                # Ensure progress bar uses a float value
                                st.progress(float(prob), text=f"**{class_names[i]}**: {prob*100:.2f}%") 

                    except Exception as e:
                        st.error(f"❌ Error saat menjalankan klasifikasi: {e}")
        else:
            st.info("⬆️ Silakan unggah gambar untuk klasifikasi RPS.")

    st.markdown('</div>', unsafe_allow_html=True)


# ===================== FOOTER DAN GARIS DEKORATIF =====================
st.markdown("""
<div style="height: 100px; background-color: #000000;"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:20px; color:#666; font-weight:600; background-color: #000000;">
🔮 Powered by Streamlit | Dibuat oleh <strong style="color:#9D8CFF;">Balqis Isaura</strong>
</div>
""", unsafe_allow_html=True)
