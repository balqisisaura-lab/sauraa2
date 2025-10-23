import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os 
from io import BytesIO
import base64

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="AI Vision - Mask Detection & Hand Gesture",
    page_icon="🤖",
    layout="wide"
)

# ========================== SOUND EFFECTS FUNCTION ==========================
def autoplay_audio_url(audio_url: str):
    """Fungsi untuk auto-play audio dari URL"""
    md = f"""
        <audio autoplay="true">
        <source src="{audio_url}" type="audio/mpeg">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# Dictionary untuk menyimpan URL audio
AUDIO_URLS = {
    # Rock Sounds
    'rock_guitar': 'https://cdn.pixabay.com/audio/2022/03/10/audio_88ef456c0d.mp3',
    'drums_heavy': 'https://cdn.pixabay.com/audio/2022/11/25/audio_24797e0017.mp3',
    'bass_drop': 'https://cdn.pixabay.com/audio/2022/03/24/audio_c3c6d6c54a.mp3',
    'crowd_cheer': 'https://cdn.pixabay.com/audio/2022/03/15/audio_01d4dd815c.mp3',
    
    # Paper Sounds
    'piano_melody': 'https://cdn.pixabay.com/audio/2022/03/23/audio_426608073c.mp3',
    'wind_chimes': 'https://cdn.pixabay.com/audio/2022/03/10/audio_72319372e4.mp3',
    'acoustic': 'https://cdn.pixabay.com/audio/2022/05/27/audio_1808fbf07a.mp3',
    'nature_ambient': 'https://cdn.pixabay.com/audio/2022/03/09/audio_62860e3805.mp3',
    
    # Scissors Sounds
    'synth_lead': 'https://cdn.pixabay.com/audio/2022/10/25/audio_25d2bbf561.mp3',
    'electronic_beat': 'https://cdn.pixabay.com/audio/2022/08/02/audio_4d49fe079e.mp3',
    'laser_sound': 'https://cdn.pixabay.com/audio/2022/03/24/audio_d1718ab41b.mp3',
    'digital_beep': 'https://cdn.pixabay.com/audio/2022/03/10/audio_d2117ba3bc.mp3',
    
    # General Sounds
    'success': 'https://cdn.pixabay.com/audio/2021/08/04/audio_12b0c7443c.mp3',
    'classify': 'https://cdn.pixabay.com/audio/2022/03/24/audio_21144ab1f3.mp3',
    'click': 'https://cdn.pixabay.com/audio/2022/03/15/audio_c2c92793d7.mp3',
    'notification': 'https://cdn.pixabay.com/audio/2022/03/12/audio_22a1b0cfc2.mp3',
    'error': 'https://cdn.pixabay.com/audio/2022/03/15/audio_c84758c1e5.mp3'
}

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@600&display=swap');

        /* === Efek Transisi Fade-In Global === */
        .stApp {
            animation: fadeInAnimation ease 0.4s; 
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }

        @keyframes fadeInAnimation {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        body, .stApp {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: #e0e0e0;
            font-family: 'Rajdhani', sans-serif;
            overflow-x: hidden;
        }

        * {
            color: #e0e0e0 !important;
        }

        .main-title {
            text-align: center;
            font-size: 5rem; 
            font-weight: 900;
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff !important;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.6); 
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: glow 2s ease-in-out infinite alternate;
            letter-spacing: 8px; 
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.6); }
            to { text-shadow: 0 0 30px rgba(0, 212, 255, 1), 0 0 60px rgba(0, 212, 255, 0.8); }
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
        
        /* CENTERED TABS NAVIGATION */
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            justify-content: center;
            gap: 1rem;
            background: rgba(30, 50, 60, 0.5);
            padding: 1rem;
            border-radius: 20px;
            margin: 2rem auto;
            max-width: 90%;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 145, 234, 0.1));
            border-radius: 15px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            transition: all 0.3s ease;
            font-family: 'Orbitron', sans-serif;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
            border-color: #00d4ff;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #00d4ff, #0091ea) !important;
            border-color: #00d4ff !important;
            box-shadow: 0 0 25px rgba(0, 212, 255, 0.6) !important;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #00d4ff, #0091ea);
            color: #fff !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            padding: 1rem 2rem !important;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
            transition: all 0.4s ease;
            font-size: 1.1rem;
            border: none;
            font-family: 'Orbitron', sans-serif;
        }
        
        .stButton > button:hover {
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.8);
            transform: translateY(-2px);
        }
        
        .card {
            background: rgba(30, 50, 60, 0.8);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
            margin-bottom: 2rem;
            border: 2px solid rgba(0, 212, 255, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .menu-item {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 145, 234, 0.1));
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            font-size: 1rem;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        
        .menu-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        }
        
        /* CUSTOM CSS FILE UPLOADER */
        .stFileUploader > div:first-child > div:first-child {
            background-color: rgba(30, 50, 60, 0.6);
            border: 3px dashed #00d4ff; 
            border-radius: 20px; 
            padding: 2rem 1.5rem; 
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2); 
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div:first-child > div:first-child:hover {
            border: 3px dashed #00ffff;
            background-color: rgba(0, 212, 255, 0.1);
        }
        
        .for-you-alert {
            padding: 1.5rem;
            border: 3px solid #ff00ff; 
            border-radius: 15px;
            background: linear-gradient(45deg, rgba(255, 0, 255, 0.1), rgba(138, 43, 226, 0.1)); 
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(255, 0, 255, 0.3);
        }

        .for-you-alert p {
            margin: 0;
            color: #ff00ff !important; 
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: #78909c !important;
            font-size: 0.9rem;
        }
        
        /* Animasi Blink untuk Success */
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        .success-blink {
            animation: blink 0.8s ease-in-out 3;
            padding: 1rem;
            border-radius: 10px;
            background: linear-gradient(45deg, rgba(0, 212, 255, 0.2), rgba(0, 255, 136, 0.2));
            border: 2px solid #00d4ff;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
            color: #00ffaa !important;
            margin-top: 1rem;
        }

        .sound-card {
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(75, 0, 130, 0.2));
            border: 2px solid rgba(138, 43, 226, 0.5);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }

        .sound-card:hover {
            transform: translateX(10px);
            box-shadow: 0 6px 20px rgba(138, 43, 226, 0.4);
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

# ========================== KONTROL STATE SAAT BERPINDAH TAB ==========================
def clear_inactive_results(current_tab_index):
    if current_tab_index != 1 and st.session_state.get('detection_result_img') is not None:
        st.session_state['detection_result_img'] = None

    if current_tab_index != 2:
        if st.session_state.get('classification_final_result') is not None:
            st.session_state['classification_final_result'] = None
        if st.session_state.get('classification_image_input') is not None:
            st.session_state['classification_image_input'] = None

# ========================== HORIZONTAL NAVIGATION (Tabs at Top - CENTERED) ==========================
tabs = st.tabs(["🏠 Beranda", "😷 Deteksi Masker", "✊✋✌ Klasifikasi Gesture", "🎵 Untuk Kamu", "ℹ Tentang"])

# ========================== MAIN CONTENT BASED ON TABS ==========================

# ----------------- BERANDA -----------------
with tabs[0]:
    clear_inactive_results(0)
    st.markdown("<h2 class='section-title'>Selamat Datang di AI Vision</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card' style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 145, 234, 0.1)); border-color: #00d4ff;'>
        <p style='font-size: 1.4rem; text-align: center;'>
            Platform <span style='font-weight: bold; color: #00d4ff;'>Computer Vision</span> berbasis AI untuk deteksi masker wajah dan pengenalan gesture tangan.
        </p>
        <p style='font-size: 1.1rem; text-align: center; font-style: italic; color: #b0bec5;'>
            Teknologi Masa Depan, Tersedia Hari Ini
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("<h3 style='text-align: center; color: #00d4ff; font-family: Orbitron, sans-serif; font-size: 2rem; margin-bottom: 1.5rem;'>Fitur Utama</h3>", unsafe_allow_html=True)
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 3rem;'>😷</p>
            <span style='font-weight: bold; color: #00d4ff;'>Deteksi Masker</span>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Deteksi apakah seseorang memakai masker atau tidak menggunakan <span style='font-weight: bold;'>YOLO</span>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat2:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 3rem;'>✊✋✌</p>
            <span style='font-weight: bold; color: #00d4ff;'>Klasifikasi Gesture</span>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Identifikasi gesture tangan: Rock, Paper, atau Scissors dengan akurasi tinggi.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat3:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 3rem;'>🎵</p>
            <span style='font-weight: bold; color: #00d4ff;'>Sound Effects</span>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Nikmati koleksi sound effects dan musik berdasarkan gesture Anda.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

# ----------------- DETEKSI MASKER -----------------
with tabs[1]:
    clear_inactive_results(1)
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

                if st.button("🔍 Deteksi Masker", type="primary", key="detect_obj"):
                    with st.spinner("⏳ Memproses deteksi masker dengan YOLO..."):
                        try:
                            results = yolo_model(image)
                            result_img = results[0].plot()  
                            result_img_rgb = Image.fromarray(result_img[..., ::-1])
                            
                            st.session_state['detection_result_img'] = result_img_rgb
                            st.markdown("<div class='success-blink'>✅ DETEKSI BERHASIL!</div>", unsafe_allow_html=True)
                            
                            # Play success sound
                            autoplay_audio_url(AUDIO_URLS['success'])
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {e}")

        with col_output_deteksi:
            if st.session_state.get('detection_result_img') is not None:
                st.image(st.session_state['detection_result_img'], caption="Hasil Deteksi YOLO", use_container_width=True)
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #00d4ff; border-radius: 15px; text-align: center; padding-top: 130px; color: #00d4ff; font-weight: bold;'>HASIL DETEKSI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠ Model YOLO tidak dapat dimuat dari '{YOLO_MODEL_PATH}'.")

# ----------------- KLASIFIKASI GESTURE -----------------
with tabs[2]:
    clear_inactive_results(2)
    st.markdown("<h2 class='section-title'>Klasifikasi Gesture Tangan ✊✋✌</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Upload gambar gesture tangan Anda! Model AI akan mengklasifikasikan apakah itu <span style='font-weight: bold; color: #00d4ff;'>Rock</span>, <span style='font-weight: bold; color: #00d4ff;'>Paper</span>, atau <span style='font-weight: bold; color: #00d4ff;'>Scissors</span>.</p>
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

                if st.button("🎯 Klasifikasikan Gesture", type="primary", key="classify_btn"):
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
                            
                            # Notifikasi sukses dengan animasi blink
                            st.markdown("<div class='success-blink'>✨ KLASIFIKASI BERHASIL!</div>", unsafe_allow_html=True)
                            
                            # Play classification sound
                            autoplay_audio_url(AUDIO_URLS['classify'])
                            
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
                
                icon = gesture_icons.get(final_result, '🤖')
                
                st.success(f"{icon} Gesture terdeteksi: **{final_result}**")
                
                st.markdown("---")
                st.markdown(f"<p style='font-size: 3rem; text-align: center;'>{icon}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 2rem; text-align: center; font-weight: bold; color: #00d4ff;'>{final_result}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #00d4ff; border-radius: 15px; text-align: center; padding-top: 130px; color: #00d4ff; font-weight: bold;'>HASIL KLASIFIKASI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)

    else:
        st.warning(f"⚠ Model Klasifikasi tidak dapat dimuat dari '{H5_MODEL_PATH}'.")

# ----------------- UNTUK KAMU (dengan Sound Effects) -----------------
with tabs[3]:
    clear_inactive_results(3)
    st.markdown("<h2 class='section-title'>🎵 Untuk Kamu - Sound Effects & Music</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Nikmati koleksi sound effects dan musik yang dipersonalisasi berdasarkan gesture Anda!</p>", unsafe_allow_html=True)
    
    current_classification = st.session_state.get('classification', 'none')
    
    # Data sound effects berdasarkan gesture
    sound_collections = {
        'rock': {
            'title': '✊ ROCK - Sound Power',
            'theme': 'Energik & Kuat',
            'color': '#ff4444',
            'sounds': [
                {'name': 'rock_guitar', 'emoji': '🎸', 'label': 'Electric Guitar Riff'},
                {'name': 'drums_heavy', 'emoji': '🥁', 'label': 'Heavy Drums Beat'},
                {'name': 'bass_drop', 'emoji': '🔊', 'label': 'Bass Drop'},
                {'name': 'crowd_cheer', 'emoji': '👏', 'label': 'Crowd Cheering'}
            ]
        },
        'paper': {
            'title': '✋ PAPER - Smooth Sounds',
            'theme': 'Lembut & Kreatif',
            'color': '#44ff44',
            'sounds': [
                {'name': 'piano_melody', 'emoji': '🎹', 'label': 'Piano Melody'},
                {'name': 'wind_chimes', 'emoji': '🎐', 'label': 'Wind Chimes'},
                {'name': 'acoustic', 'emoji': '🎻', 'label': 'Acoustic Strings'},
                {'name': 'nature_ambient', 'emoji': '🌿', 'label': 'Nature Ambient'}
            ]
        },
        'scissors': {
            'title': '✌ SCISSORS - Sharp Beats',
            'theme': 'Tajam & Presisi',
            'color': '#ffaa00',
            'sounds': [
                {'name': 'synth_lead', 'emoji': '🎛️', 'label': 'Synth Lead'},
                {'name': 'electronic_beat', 'emoji': '⚡', 'label': 'Electronic Beat'},
                {'name': 'laser_sound', 'emoji': '🔫', 'label': 'Laser Effect'},
                {'name': 'digital_beep', 'emoji': '📟', 'label': 'Digital Beeps'}
            ]
        }
    }
    
    if current_classification in sound_collections:
        collection = sound_collections[current_classification]
        
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(45deg, rgba({collection['color'][1:3]}, {collection['color'][3:5]}, {collection['color'][5:7]}, 0.2), rgba(138, 43, 226, 0.2)); border-color: {collection['color']};'>
            <p style='font-size: 2rem; text-align: center; color: {collection['color']}; font-weight: bold;'>{collection['title']}</p>
            <p style='font-size: 1.2rem; text-align: center; font-style: italic;'>Tema: {collection['theme']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎧 Koleksi Sound Effects")
        
        col_sound1, col_sound2 = st.columns(2)
        
        for idx, sound in enumerate(collection['sounds']):
            with col_sound1 if idx % 2 == 0 else col_sound2:
                st.markdown(f"""
                <div class='sound-card'>
                    <p style='font-size: 2rem; text-align: center; margin: 0;'>{sound['emoji']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"▶️ {sound['label']}", key=f"play_{sound['name']}_{current_classification}"):
                    autoplay_audio_url(AUDIO_URLS[sound['name']])
                    st.success(f"🔊 Now Playing: {sound['label']}")
        
        st.markdown("---")
        
        # Playlist Rekomendasi
        st.markdown("### 🎶 Playlist Rekomendasi")
        
        playlists = {
            'rock': [
                "Rock Anthems - High Energy Mix",
                "Metal Madness - Headbanging Collection",
                "Power Chords - Guitar Heroes"
            ],
            'paper': [
                "Chill Vibes - Relaxation Mix",
                "Creative Flow - Focus Music",
                "Ambient Dreams - Peaceful Sounds"
            ],
            'scissors': [
                "Electronic Dance - EDM Hits",
                "Future Bass - Modern Beats",
                "Synthwave - Retro Future"
            ]
        }
        
        if current_classification in playlists:
            for playlist in playlists[current_classification]:
                st.markdown(f"""
                <div class='menu-item'>
                    <span style='font-size: 1.1rem;'>🎵 {playlist}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Fun Facts
        st.markdown("---")
        st.markdown("### 💡 Fun Fact")
        
        fun_facts = {
            'rock': "Musik rock dapat meningkatkan motivasi dan energi hingga 40% saat berolahraga!",
            'paper': "Musik ambient dan piano dapat meningkatkan kreativitas dan produktivitas kerja.",
            'scissors': "Beat elektronik dengan tempo cepat dapat meningkatkan fokus dan konsentrasi."
        }
        
        st.info(f"🎵 {fun_facts.get(current_classification, 'Musik memiliki kekuatan luar biasa!')}")
        
    else:
        st.markdown("""
        <div class='for-you-alert'>
            <p>
                <span style='font-size: 2rem;'>🎵</span>
                <br>
                <span style='font-weight: bold;'>Lakukan Klasifikasi Gesture terlebih dahulu</span>
                <br>
                untuk mendapatkan koleksi sound effects yang dipersonalisasi untukmu!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # General Sound Effects
        st.markdown("---")
        st.markdown("### 🎵 Sound Effects Umum")
        
        general_sounds = [
            {'name': 'click', 'emoji': '🖱️', 'label': 'Click Sound'},
            {'name': 'notification', 'emoji': '🔔', 'label': 'Notification Bell'},
            {'name': 'success', 'emoji': '✅', 'label': 'Success Sound'},
            {'name': 'error', 'emoji': '❌', 'label': 'Error Buzz'}
        ]
        
        col_gen1, col_gen2 = st.columns(2)
        
        for idx, sound in enumerate(general_sounds):
            with col_gen1 if idx % 2 == 0 else col_gen2:
                st.markdown(f"""
                <div class='sound-card'>
                    <p style='font-size: 2rem; text-align: center; margin: 0;'>{sound['emoji']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"▶️ {sound['label']}", key=f"general_{sound['name']}"):
                    autoplay_audio_url(AUDIO_URLS[sound['name']])
                    st.success(f"🔊 Now Playing: {sound['label']}")

# ----------------- KONTAK -----------------
with tabs[4]:
    clear_inactive_results(4)
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
with tabs[5]:
    clear_inactive_results(5)
    st.markdown("<h2 class='section-title'>Tentang AI Vision ℹ</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.3rem;'>AI Vision adalah platform Computer Vision yang menggabungkan teknologi <span style='font-weight: bold; color: #00d4ff;'>Object Detection</span> dan <span style='font-weight: bold; color: #00d4ff;'>Image Classification</span>.</p>
        <p style='margin-top: 1rem;'>Teknologi yang digunakan:</p>
        <ul>
            <li><span style='font-weight: bold; color: #00d4ff;'>YOLO (You Only Look Once):</span> Untuk deteksi masker wajah secara real-time</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Deep Learning (H5 Model):</span> Untuk klasifikasi gesture tangan (Rock, Paper, Scissors)</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Streamlit:</span> Framework untuk antarmuka web yang interaktif</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Sound Effects:</span> Audio personalisasi berdasarkan gesture</li>
        </ul>
        <p style='margin-top: 1.5rem;'>Dikembangkan oleh <span style='font-weight: bold; color: #00d4ff;'>Balqis Isaura</span></p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #00d4ff;'>#ComputerVision #AI #MachineLearning #SoundEffects</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2024 AI Vision. Dibuat dengan 🤖 dan ❤ oleh Balqis Isaura.</p>", unsafe_allow_html=True)
