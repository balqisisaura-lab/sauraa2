/* Warna teks yang jelas */
        body, p, span, div, label, h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0;
            font-family: 'Rajdhani', sans-serif;
        }import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os 
from io import BytesIO

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="AI Vision - Balqis Isaura",
    page_icon="🤖",
    layout="wide"
)

# ========================== CUSTOM STYLE WITH ANIMATED BACKGROUND ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@600&display=swap');

        /* Background dengan animasi */
        body {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            animation: gradient-animation 15s ease infinite;
            background-size: 400% 400%;
        }
        
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stApp {
            background: transparent;
        }

        .main-title {
            text-align: center;
            font-size: 5rem; 
            font-weight: 900;
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff !important;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.6), 0 0 60px rgba(138, 43, 226, 0.4); 
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: glow 2s ease-in-out infinite alternate;
            letter-spacing: 8px; 
        }
        
        @keyframes glow {
            from { 
                text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.6), 0 0 60px rgba(138, 43, 226, 0.4); 
            }
            to { 
                text-shadow: 0 0 30px rgba(0, 212, 255, 1), 0 0 60px rgba(0, 212, 255, 0.8), 0 0 90px rgba(138, 43, 226, 0.6); 
            }
        }
        
        .subtitle {
            text-align: center;
            color: #b0bec5 !important;
            font-size: 1.5rem;
            margin-bottom: 2rem;
            font-style: italic;
        }
        
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #00d4ff !important;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
            margin-top: 2rem;
            text-align: center;
            font-family: 'Orbitron', sans-serif;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #00d4ff, #8a2be2, #ff00ff);
            background-size: 200% 200%;
            color: #fff !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            padding: 1rem 2rem !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            transition: all 0.4s ease;
            font-size: 1.1rem;
            border: none;
            font-family: 'Orbitron', sans-serif;
            animation: gradient-shift 3s ease infinite;
        }
        
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .stButton > button:hover {
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.8), 0 0 50px rgba(138, 43, 226, 0.6);
            transform: translateY(-2px) scale(1.02);
        }
        
        .card {
            background: rgba(30, 50, 60, 0.7);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
            margin-bottom: 2rem;
            border: 2px solid rgba(0, 212, 255, 0.3);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: rgba(138, 43, 226, 0.5);
            box-shadow: 0 12px 40px rgba(138, 43, 226, 0.3);
        }
        
        .menu-item {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(138, 43, 226, 0.1));
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            font-size: 1rem;
            border: 1px solid rgba(0, 212, 255, 0.3);
            backdrop-filter: blur(5px);
        }
        
        .menu-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(138, 43, 226, 0.4);
            border-color: rgba(138, 43, 226, 0.5);
        }
        
        /* CUSTOM CSS FILE UPLOADER */
        .stFileUploader > div:first-child > div:first-child {
            background-color: rgba(30, 50, 60, 0.6);
            border: 3px dashed #00d4ff; 
            border-radius: 20px; 
            padding: 2rem 1.5rem; 
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2); 
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stFileUploader > div:first-child > div:first-child:hover {
            border: 3px dashed #8a2be2;
            background-color: rgba(138, 43, 226, 0.1);
            box-shadow: 0 6px 20px rgba(138, 43, 226, 0.3);
        }
        
        .recommendation-alert {
            padding: 1.5rem;
            border: 3px solid #ffd700; 
            border-radius: 15px;
            background: linear-gradient(45deg, rgba(255, 215, 0, 0.1), rgba(255, 193, 7, 0.1)); 
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
            backdrop-filter: blur(10px);
        }

        .recommendation-alert p {
            margin: 0;
            color: #ffd700 !important; 
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: #78909c !important;
            font-size: 0.9rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 32, 39, 0.95) 0%, rgba(32, 58, 67, 0.95) 100%);
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            margin-bottom: 0.5rem;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(0, 212, 255, 0.2);
            border-color: #00d4ff;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>AI VISION</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Deteksi Masker Wajah & Klasifikasi Gesture Tangan dengan Kecerdasan Buatan</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== INITIALIZE SESSION STATE ==========================
if 'classification' not in st.session_state:
    st.session_state['classification'] = 'none'

if 'detection_result_img' not in st.session_state:
    st.session_state['detection_result_img'] = None
if 'classification_final_result' not in st.session_state:
    st.session_state['classification_final_result'] = None
if 'classification_image_input' not in st.session_state:
    st.session_state['classification_image_input'] = None

if 'last_yolo_uploader' not in st.session_state:
    st.session_state['last_yolo_uploader'] = None
if 'last_classify_uploader' not in st.session_state:
    st.session_state['last_classify_uploader'] = None

# State untuk navigasi
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

# ========================== UTILITY FUNCTIONS (Load Models) ==========================
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO dari '{path}'. Error: {e}")
        return None

@st.cache_resource
def load_classification_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model Klasifikasi H5 dari '{path}'. Error: {e}")
        return None

# ========================== KONTROL STATE SAAT BERPINDAH HALAMAN ==========================
def clear_inactive_results(current_tab_index):
    if current_tab_index != 'mask' and st.session_state.get('detection_result_img') is not None:
        st.session_state['detection_result_img'] = None

    if current_tab_index != 'gesture':
        if st.session_state.get('classification_final_result') is not None:
            st.session_state['classification_final_result'] = None
        if st.session_state.get('classification_image_input') is not None:
            st.session_state['classification_image_input'] = None

# ========================== NAVIGATION MENU ==========================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00d4ff;'>📱 Menu Navigasi</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("🏠 Beranda", use_container_width=True):
        st.session_state['current_page'] = 'home'
        st.rerun()
    
    if st.button("😷 Deteksi Masker", use_container_width=True):
        st.session_state['current_page'] = 'mask'
        st.rerun()
    
    if st.button("✊✋✌ Klasifikasi Gesture", use_container_width=True):
        st.session_state['current_page'] = 'gesture'
        st.rerun()
    
    if st.button("🎯 Rekomendasi", use_container_width=True):
        st.session_state['current_page'] = 'recommendation'
        st.rerun()
    
    if st.button("📞 Kontak", use_container_width=True):
        st.session_state['current_page'] = 'contact'
        st.rerun()
    
    if st.button("ℹ Tentang", use_container_width=True):
        st.session_state['current_page'] = 'about'
        st.rerun()

# ========================== MAIN CONTENT BASED ON CURRENT PAGE ==========================
current_page = st.session_state.get('current_page', 'home')

# ----------------- BERANDA -----------------
if current_page == 'home':
    clear_inactive_results('home')
    st.markdown("<h2 class='section-title'>Selamat Datang di AI Vision</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(138, 43, 226, 0.15)); border-color: #00d4ff;'>
        <p style='font-size: 1.4rem; text-align: center;'>
            Platform <span style='font-weight: bold; color: #00d4ff;'>Computer Vision</span> berbasis AI untuk deteksi masker wajah dan pengenalan gesture tangan.
        </p>
        <p style='font-size: 1.1rem; text-align: center; font-style: italic; color: #b0bec5;'>
            Teknologi Masa Depan, Tersedia Hari Ini
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("<h3 style='text-align: center; color: #00d4ff; font-family: Orbitron, sans-serif; font-size: 2rem; margin-bottom: 2rem;'>Fitur Utama</h3>", unsafe_allow_html=True)
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        <div class='menu-item' style='min-height: 280px; padding: 2.5rem;'>
            <p style='font-size: 5rem; margin-bottom: 1rem;'>😷</p>
            <span style='font-weight: bold; color: #00d4ff; font-size: 1.8rem;'>Deteksi Masker</span>
            <p style='font-size: 1.05rem; margin-top: 1.2rem; line-height: 1.7;'>Deteksi apakah seseorang memakai masker atau tidak menggunakan <span style='font-weight: bold; color: #8a2be2;'>YOLO</span>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Mulai Deteksi Masker", use_container_width=True, type="primary", key="btn_mask"):
            st.session_state['current_page'] = 'mask'
            st.rerun()

    with col_feat2:
        st.markdown("""
        <div class='menu-item' style='min-height: 280px; padding: 2.5rem;'>
            <p style='font-size: 5rem; margin-bottom: 1rem;'>✊✋✌</p>
            <span style='font-weight: bold; color: #00d4ff; font-size: 1.8rem;'>Klasifikasi Gesture</span>
            <p style='font-size: 1.05rem; margin-top: 1.2rem; line-height: 1.7;'>Identifikasi gesture tangan: Rock, Paper, atau Scissors dengan akurasi tinggi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Mulai Klasifikasi Gesture", use_container_width=True, type="primary", key="btn_gesture"):
            st.session_state['current_page'] = 'gesture'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

# ----------------- DETEKSI MASKER -----------------
elif current_page == 'mask':
    clear_inactive_results('mask')
    st.markdown("<h2 class='section-title'>Deteksi Masker Wajah 😷</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Menggunakan model <span style='font-weight: bold; color: #00d4ff;'>YOLO</span> untuk mendeteksi apakah seseorang memakai masker atau tidak. Upload gambar wajah dan lihat hasilnya!</p>
    </div>
    """, unsafe_allow_html=True)
    
    YOLO_MODEL_PATH = 'model/Balqis Isaura_Laporan 4.pt'
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    
    if yolo_model:
        col_input_deteksi, col_output_deteksi = st.columns(2) 

        with col_input_deteksi:
            uploaded_file_deteksi = st.file_uploader("Upload Gambar Wajah (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="yolo_uploader")
            
            if st.session_state.get('last_yolo_uploader') != uploaded_file_deteksi:
                st.session_state['detection_result_img'] = None
                st.session_state['last_yolo_uploader'] = uploaded_file_deteksi

            if uploaded_file_deteksi:
                image = Image.open(uploaded_file_deteksi)
                st.image(image, caption="Gambar Input", use_container_width=True)

                if st.button("🔍 Deteksi Masker Sekarang", type="primary", key="detect_obj"):
                    with st.spinner("⏳ Memproses deteksi masker dengan YOLO..."):
                        try:
                            results = yolo_model(image)
                            result_img = results[0].plot()  
                            result_img_rgb = Image.fromarray(result_img[..., ::-1])
                            
                            st.session_state['detection_result_img'] = result_img_rgb
                            st.success("✅ Deteksi berhasil! Hasil telah ditampilkan.")
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {e}")

        with col_output_deteksi:
            if st.session_state.get('detection_result_img') is not None:
                st.image(st.session_state['detection_result_img'], caption="Hasil Deteksi YOLO", use_container_width=True)
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #00d4ff; border-radius: 15px; text-align: center; padding-top: 130px; color: #00d4ff; font-weight: bold; backdrop-filter: blur(5px); background: rgba(0, 212, 255, 0.05);'>HASIL DETEKSI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Model YOLO tidak dapat dimuat dari '{YOLO_MODEL_PATH}'.")

# ----------------- KLASIFIKASI GESTURE -----------------
elif current_page == 'gesture':
    clear_inactive_results('gesture')
    st.markdown("<h2 class='section-title'>Klasifikasi Gesture Tangan ✊✋✌</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Upload gambar gesture tangan Anda! Model AI akan mengklasifikasikan apakah itu <span style='font-weight: bold; color: #00d4ff;'>Rock</span>, <span style='font-weight: bold; color: #8a2be2;'>Paper</span>, atau <span style='font-weight: bold; color: #ff00ff;'>Scissors</span>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    H5_MODEL_PATH = 'model/compressed.h5'
    classification_model = load_classification_model(H5_MODEL_PATH)
    
    if classification_model:
        col_class_input, col_class_output = st.columns(2)

        with col_class_input:
            uploaded_file_class = st.file_uploader("Upload Gambar Gesture (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="classify_uploader")

            if st.session_state.get('last_classify_uploader') != uploaded_file_class:
                st.session_state['classification_final_result'] = None
                st.session_state['classification_image_input'] = None
                st.session_state['last_classify_uploader'] = uploaded_file_class

            if uploaded_file_class:
                image_pil = Image.open(uploaded_file_class)
                
                # Convert ke RGB (3 channel)
                image_rgb = image_pil.convert('RGB')
                
                # Resize ke 128x128 sesuai yang diharapkan model
                image_class_resized = image_rgb.resize((128, 128))
                
                st.session_state['classification_image_input'] = image_class_resized
                st.image(st.session_state['classification_image_input'], caption="Gambar Input (128x128 RGB)", use_container_width=True)

                if st.button("🎯 Klasifikasikan Gesture Sekarang", type="primary", key="classify_btn"):
                    with st.spinner("⏳ Mengklasifikasikan gesture dengan AI..."):
                        try:
                            img_array = np.array(image_class_resized)
                            
                            # Normalisasi pixel values ke [0, 1]
                            img_array = img_array / 255.0
                            
                            # Expand dimensions untuk batch (1, 128, 128, 3)
                            preprocessed_img = np.expand_dims(img_array, axis=0)
                            
                            # Prediksi
                            predictions = classification_model.predict(preprocessed_img)
                            
                            # Class names untuk Rock Paper Scissors
                            class_names = ['Rock', 'Paper', 'Scissors']
                            
                            # Dapatkan index dengan confidence tertinggi
                            predicted_class_idx = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class_idx] * 100
                            
                            final_result = class_names[predicted_class_idx]
                            
                            # Simpan ke session state
                            st.session_state['classification_final_result'] = final_result
                            st.session_state['classification'] = final_result.lower()
                            
                            # Efek visual
                            if final_result == 'Rock':
                                st.balloons()
                            elif final_result == 'Paper':
                                st.snow()
                            else:
                                st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {e}")

        with col_class_output:
            if st.session_state.get('classification_final_result') is not None:
                final_result = st.session_state['classification_final_result']
                
                st.markdown("### 🎯 Hasil Klasifikasi AI")
                
                # Icon untuk setiap gesture
                gesture_icons = {
                    'Rock': '✊',
                    'Paper': '✋',
                    'Scissors': '✌'
                }
                
                gesture_colors = {
                    'Rock': '#00d4ff',
                    'Paper': '#8a2be2',
                    'Scissors': '#ff00ff'
                }
                
                icon = gesture_icons.get(final_result, '🤖')
                color = gesture_colors.get(final_result, '#00d4ff')
                
                st.success(f"{icon} Gesture terdeteksi: **{final_result}**")
                
                st.markdown("---")
                st.markdown(f"<p style='font-size: 4rem; text-align: center;'>{icon}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 2.5rem; text-align: center; font-weight: bold; color: {color};'>{final_result}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #00d4ff; border-radius: 15px; text-align: center; padding-top: 130px; color: #00d4ff; font-weight: bold; backdrop-filter: blur(5px); background: rgba(0, 212, 255, 0.05);'>HASIL KLASIFIKASI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)

    else:
        st.warning(f"⚠️ Model Klasifikasi tidak dapat dimuat dari '{H5_MODEL_PATH}'.")

# ----------------- REKOMENDASI -----------------
elif current_page == 'recommendation':
    clear_inactive_results('recommendation')
    st.markdown("<h2 class='section-title'>Rekomendasi Berdasarkan Gesture 🎯</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rekomendasi ini didasarkan pada hasil klasifikasi gesture Anda.</p>", unsafe_allow_html=True)
    
    recommendations = {
        'rock': {
            'title': '✊ ROCK - Kekuatan & Ketegasan',
            'color': '#00d4ff',
            'items': [
                {'nama': 'Action Games', 'deskripsi': 'Game penuh aksi dan tantangan yang membutuhkan ketegasan.', 'kategori': 'Gaming'},
                {'nama': 'Strength Training', 'deskripsi': 'Program latihan kekuatan untuk membangun otot.', 'kategori': 'Fitness'}
            ]
        },
        'paper': {
            'title': '✋ PAPER - Fleksibilitas & Kreativitas',
            'color': '#8a2be2',
            'items': [
                {'nama': 'Creative Writing', 'deskripsi': 'Kursus menulis kreatif untuk mengembangkan imajinasi.', 'kategori': 'Education'},
                {'nama': 'Digital Art', 'deskripsi': 'Belajar seni digital dan desain grafis.', 'kategori': 'Art'}
            ]
        },
        'scissors': {
            'title': '✌ SCISSORS - Ketepatan & Strategi',
            'color': '#ff00ff',
            'items': [
                {'nama': 'Strategy Games', 'deskripsi': 'Game strategi yang melatih pemikiran taktis.', 'kategori': 'Gaming'},
                {'nama': 'Chess Lessons', 'deskripsi': 'Pelajari strategi catur dari master.', 'kategori': 'Education'}
            ]
        }
    }
    
    col_rec1, col_rec2 = st.columns(2)
    
    current_classification = st.session_state.get('classification', 'none')
    
    if current_classification in recommendations:
        rec_data = recommendations[current_classification]
        
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(45deg, rgba(0, 212, 255, 0.2), rgba(138, 43, 226, 0.2)); border-color: {rec_data['color']};'>
            <p style='font-size: 1.8rem; text-align: center; color: {rec_data['color']}; font-weight: bold;'>{rec_data['title']}</p>
            <p style='font-size: 1.1rem; text-align: center;'>Berdasarkan gesture Anda, kami merekomendasikan:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🎮 Rekomendasi Utama")
            for item in rec_data['items']:
                st.markdown(f"""
                <div class='menu-item'>
                    <span style='font-weight: bold; color: {rec_data['color']};'>{item['nama']}</span>
                    <br>
                    <span style='font-size: 0.85rem; color: #ffd700;'>[{item['kategori']}]</span>
                    <br>
                    <span style='font-size: 0.9rem;'>{item['deskripsi']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("### 💡 Tips & Saran")
            tips = {
                'rock': ['Fokus pada kekuatan mental', 'Bangun ketahanan', 'Latih konsistensi'],
                'paper': ['Eksplorasi ide baru', 'Berpikir out of the box', 'Fleksibel dalam pendekatan'],
                'scissors': ['Rencanakan strategi', 'Analisa sebelum bertindak', 'Fokus pada detail']
            }
            
            for tip in tips.get(current_classification, []):
                st.markdown(f"""
                <div class='menu-item'>
                    <span style='font-size: 1rem;'>💡 {tip}</span>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class='recommendation-alert'>
            <p>
                <span style='font-size: 2rem;'>✊✋✌</span>
                <br>
                <span style='font-weight: bold;'>Silakan lakukan Klasifikasi Gesture terlebih dahulu</span> untuk mendapatkan rekomendasi personal.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- KONTAK -----------------
elif current_page == 'contact':
    clear_inactive_results('contact')
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Punya pertanyaan atau feedback? Hubungi tim kami!</p>
        <div style='margin-top: 1.5rem;'>
            <p><span style='font-weight: bold; color: #00d4ff;'>📍 Alamat:</span> Jl. AI Technology No. 42, Tech City</p>
            <p><span style='font-weight: bold; color: #00d4ff;'>📞 Telepon:</span> (021) 555-VISION</p>
            <p><span style='font-weight: bold; color: #00d4ff;'>📧 Email:</span> <a href='mailto:contact@aivision.ai' style='color: #00d4ff !important; text-decoration: none;'>contact@aivision.ai</a></p>
            <p><span style='font-weight: bold; color: #00d4ff;'>🕒 Support:</span> 24/7 Online</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- TENTANG -----------------
elif current_page == 'about':
    clear_inactive_results('about')
    st.markdown("<h2 class='section-title'>Tentang AI Vision ℹ</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.3rem;'>AI Vision adalah platform Computer Vision yang menggabungkan teknologi <span style='font-weight: bold; color: #00d4ff;'>Object Detection</span> dan <span style='font-weight: bold; color: #8a2be2;'>Image Classification</span>.</p>
        <p style='margin-top: 1rem;'>Teknologi yang digunakan:</p>
        <ul>
            <li><span style='font-weight: bold; color: #00d4ff;'>YOLO (You Only Look Once):</span> Untuk deteksi masker wajah secara real-time</li>
            <li><span style='font-weight: bold; color: #8a2be2;'>Deep Learning (H5 Model):</span> Untuk klasifikasi gesture tangan (Rock, Paper, Scissors)</li>
            <li><span style='font-weight: bold; color: #ff00ff;'>Streamlit:</span> Framework untuk antarmuka web yang interaktif</li>
        </ul>
        <p style='margin-top: 1.5rem;'>Dikembangkan oleh <span style='font-weight: bold; color: #00d4ff;'>Balqis Isaura</span></p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #8a2be2;'>#ComputerVision #AI #MachineLearning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2024 AI Vision. Dibuat dengan 🤖 dan ❤ oleh Balqis Isaura.</p>", unsafe_allow_html=True)
