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
    # Sound effects untuk deteksi & klasifikasi
    'success': 'https://cdn.pixabay.com/audio/2021/08/04/audio_12b0c7443c.mp3',
    'classify': 'https://cdn.pixabay.com/audio/2022/03/24/audio_21144ab1f3.mp3',
    'rock_detect': 'https://cdn.pixabay.com/audio/2022/03/10/audio_88ef456c0d.mp3',  # Rock sound
    'paper_detect': 'https://cdn.pixabay.com/audio/2022/03/23/audio_426608073c.mp3',  # Paper sound
    'scissors_detect': 'https://cdn.pixabay.com/audio/2022/10/25/audio_25d2bbf561.mp3',  # Scissors sound
}

# Dictionary untuk rekomendasi lagu berdasarkan gesture
MUSIC_RECOMMENDATIONS = {
    'rock': {
        'genre': 'Rock & Metal',
        'color': '#ff4444',
        'emoji': '✊',
        'songs': [
            {'title': 'Bohemian Rhapsody', 'artist': 'Queen', 'year': '1975'},
            {'title': 'Stairway to Heaven', 'artist': 'Led Zeppelin', 'year': '1971'},
            {'title': 'Sweet Child O\' Mine', 'artist': 'Guns N\' Roses', 'year': '1987'},
            {'title': 'Smells Like Teen Spirit', 'artist': 'Nirvana', 'year': '1991'},
            {'title': 'Enter Sandman', 'artist': 'Metallica', 'year': '1991'},
            {'title': 'Highway to Hell', 'artist': 'AC/DC', 'year': '1979'},
            {'title': 'Paranoid', 'artist': 'Black Sabbath', 'year': '1970'},
            {'title': 'Back in Black', 'artist': 'AC/DC', 'year': '1980'},
        ],
        'playlists': [
            '🎸 Rock Classics - Timeless Anthems',
            '⚡ Metal Madness - Heavy Hits',
            '🔥 Hard Rock Heroes - Power Collection',
            '🎵 90s Rock Revival - Grunge & Alternative'
        ]
    },
    'paper': {
        'genre': 'Jazz & Classical',
        'color': '#44ff44',
        'emoji': '✋',
        'songs': [
            {'title': 'Take Five', 'artist': 'Dave Brubeck', 'year': '1959'},
            {'title': 'Autumn Leaves', 'artist': 'Bill Evans', 'year': '1959'},
            {'title': 'Clair de Lune', 'artist': 'Claude Debussy', 'year': '1905'},
            {'title': 'The Girl from Ipanema', 'artist': 'Stan Getz', 'year': '1964'},
            {'title': 'My Funny Valentine', 'artist': 'Chet Baker', 'year': '1954'},
            {'title': 'Four Seasons - Spring', 'artist': 'Vivaldi', 'year': '1725'},
            {'title': 'Fly Me to the Moon', 'artist': 'Frank Sinatra', 'year': '1964'},
            {'title': 'Blue in Green', 'artist': 'Miles Davis', 'year': '1959'},
        ],
        'playlists': [
            '🎹 Jazz Essentials - Smooth & Sophisticated',
            '🎻 Classical Masterpieces - Timeless Beauty',
            '🎺 Chill Jazz - Coffee Shop Vibes',
            '🎼 Piano & Strings - Peaceful Melodies'
        ]
    },
    'scissors': {
        'genre': 'Electronic & Pop',
        'color': '#ffaa00',
        'emoji': '✌',
        'songs': [
            {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'year': '2011'},
            {'title': 'Levels', 'artist': 'Avicii', 'year': '2011'},
            {'title': 'Strobe', 'artist': 'Deadmau5', 'year': '2009'},
            {'title': 'Animals', 'artist': 'Martin Garrix', 'year': '2013'},
            {'title': 'Blinding Lights', 'artist': 'The Weeknd', 'year': '2019'},
            {'title': 'One More Time', 'artist': 'Daft Punk', 'year': '2000'},
            {'title': 'Wake Me Up', 'artist': 'Avicii', 'year': '2013'},
            {'title': 'Midnight City', 'artist': 'M83', 'year': '2011'},
        ],
        'playlists': [
            '⚡ EDM Bangers - Festival Anthems',
            '🎛️ Electronic Dreams - Synthwave Mix',
            '💫 Pop Hits - Chart Toppers',
            '🌟 Future Bass - Modern Beats'
        ]
    }
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
            position: relative;
        }

        /* Animated Background Particles */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(2px 2px at 20% 30%, rgba(0, 212, 255, 0.4), transparent),
                radial-gradient(2px 2px at 60% 70%, rgba(138, 43, 226, 0.4), transparent),
                radial-gradient(1px 1px at 50% 50%, rgba(0, 212, 255, 0.3), transparent),
                radial-gradient(1px 1px at 80% 10%, rgba(255, 0, 255, 0.3), transparent),
                radial-gradient(2px 2px at 90% 60%, rgba(0, 212, 255, 0.4), transparent),
                radial-gradient(1px 1px at 33% 80%, rgba(138, 43, 226, 0.3), transparent);
            background-size: 200% 200%;
            animation: particleFloat 20s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }

        @keyframes particleFloat {
            0%, 100% { 
                background-position: 0% 0%, 100% 100%, 50% 50%, 80% 10%, 90% 60%, 33% 80%;
                opacity: 1;
            }
            25% { 
                background-position: 100% 50%, 0% 50%, 25% 75%, 60% 30%, 70% 80%, 50% 60%;
                opacity: 0.8;
            }
            50% { 
                background-position: 50% 100%, 50% 0%, 75% 25%, 40% 50%, 50% 40%, 70% 40%;
                opacity: 1;
            }
            75% { 
                background-position: 0% 50%, 100% 50%, 60% 60%, 20% 70%, 30% 20%, 40% 70%;
                opacity: 0.8;
            }
        }

        /* Falling Glitter Effect */
        .stApp::after {
            content: '';
            position: fixed;
            top: -50%;
            left: 0;
            width: 100%;
            height: 150%;
            background-image: 
                radial-gradient(1px 1px at 10% 10%, rgba(0, 212, 255, 0.6), transparent),
                radial-gradient(1px 1px at 20% 30%, rgba(255, 255, 255, 0.4), transparent),
                radial-gradient(1px 1px at 30% 50%, rgba(0, 212, 255, 0.5), transparent),
                radial-gradient(1px 1px at 40% 20%, rgba(138, 43, 226, 0.5), transparent),
                radial-gradient(1px 1px at 50% 40%, rgba(255, 255, 255, 0.3), transparent),
                radial-gradient(1px 1px at 60% 60%, rgba(0, 212, 255, 0.4), transparent),
                radial-gradient(1px 1px at 70% 30%, rgba(255, 0, 255, 0.4), transparent),
                radial-gradient(1px 1px at 80% 50%, rgba(255, 255, 255, 0.5), transparent),
                radial-gradient(1px 1px at 90% 70%, rgba(0, 212, 255, 0.6), transparent),
                radial-gradient(1px 1px at 15% 80%, rgba(138, 43, 226, 0.4), transparent);
            background-size: 100% 100%;
            animation: glitterFall 15s linear infinite;
            pointer-events: none;
            z-index: 0;
            opacity: 0.6;
        }

        @keyframes glitterFall {
            0% {
                transform: translateY(-50%);
            }
            100% {
                transform: translateY(100%);
            }
        }

        /* Floating Orbs */
        @keyframes floatOrb {
            0%, 100% {
                transform: translate(0, 0) scale(1);
                opacity: 0.3;
            }
            33% {
                transform: translate(30px, -50px) scale(1.1);
                opacity: 0.5;
            }
            66% {
                transform: translate(-20px, -30px) scale(0.9);
                opacity: 0.4;
            }
        }

        /* Make sure content stays above background */
        .stApp > div {
            position: relative;
            z-index: 1;
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
            animation: slideInUp 0.5s ease-out;
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
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
            transform: scale(1.02);
        }
        
        .for-you-alert {
            padding: 1.5rem;
            border: 3px solid #ff00ff; 
            border-radius: 15px;
            background: linear-gradient(45deg, rgba(255, 0, 255, 0.1), rgba(138, 43, 226, 0.1)); 
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(255, 0, 255, 0.3);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
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

        .song-card {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(138, 43, 226, 0.1));
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 15px;
            padding: 1.2rem;
            margin: 0.8rem 0;
            transition: all 0.3s ease;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .song-card:hover {
            transform: translateX(10px) scale(1.02);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
            border-color: #00d4ff;
        }

        .playlist-card {
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.15), rgba(75, 0, 130, 0.15));
            border: 2px solid rgba(138, 43, 226, 0.4);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.6rem 0;
            transition: all 0.3s ease;
        }

        .playlist-card:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 15px rgba(138, 43, 226, 0.5);
        }

        /* Animasi untuk gambar */
        .stImage {
            transition: transform 0.3s ease;
        }

        .stImage:hover {
            transform: scale(1.02);
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
            <span style='font-weight: bold; color: #00d4ff;'>Music Recommendations</span>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Dapatkan rekomendasi musik yang dipersonalisasi berdasarkan gesture Anda.</p>
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
                            
                            # Play sound sesuai gesture yang terdeteksi
                            gesture_lower = final_result.lower()
                            if gesture_lower == 'rock':
                                autoplay_audio_url(AUDIO_URLS['rock_detect'])
                            elif gesture_lower == 'paper':
                                autoplay_audio_url(AUDIO_URLS['paper_detect'])
                            elif gesture_lower == 'scissors':
                                autoplay_audio_url(AUDIO_URLS['scissors_detect'])
                            
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
                
                # Tampilkan info musik yang akan direkomendasikan
                gesture_lower = final_result.lower()
                if gesture_lower in MUSIC_RECOMMENDATIONS:
                    music_genre = MUSIC_RECOMMENDATIONS[gesture_lower]['genre']
                    st.markdown("---")
                    st.info(f"🎵 Cek tab 'Untuk Kamu' untuk rekomendasi musik **{music_genre}**!")
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #00d4ff; border-radius: 15px; text-align: center; padding-top: 130px; color: #00d4ff; font-weight: bold;'>HASIL KLASIFIKASI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)

    else:
        st.warning(f"⚠ Model Klasifikasi tidak dapat dimuat dari '{H5_MODEL_PATH}'.")

# ----------------- UNTUK KAMU (Music Recommendations) -----------------
with tabs[3]:
    clear_inactive_results(3)
    st.markdown("<h2 class='section-title'>🎵 Untuk Kamu - Music Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Dapatkan rekomendasi musik yang dipersonalisasi berdasarkan gesture Anda!</p>", unsafe_allow_html=True)
    
    current_classification = st.session_state.get('classification', 'none')
    
    if current_classification in MUSIC_RECOMMENDATIONS:
        music_data = MUSIC_RECOMMENDATIONS[current_classification]
        
        # Header dengan genre
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(45deg, rgba({int(music_data['color'][1:3], 16)}, {int(music_data['color'][3:5], 16)}, {int(music_data['color'][5:7], 16)}, 0.2), rgba(138, 43, 226, 0.2)); border-color: {music_data['color']};'>
            <p style='font-size: 3rem; text-align: center;'>{music_data['emoji']}</p>
            <p style='font-size: 2rem; text-align: center; color: {music_data['color']}; font-weight: bold;'>{music_data['genre']}</p>
            <p style='font-size: 1.2rem; text-align: center; font-style: italic;'>Rekomendasi Spesial untuk Gesture <span style='font-weight: bold; color: {music_data['color']};'>{current_classification.upper()}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Section: Top Recommended Songs
        st.markdown("### 🎧 Top Recommended Songs")
        st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #b0bec5;'>Lagu-lagu pilihan yang cocok dengan vibe gesture Anda</p>", unsafe_allow_html=True)
        
        col_song1, col_song2 = st.columns(2)
        
        for idx, song in enumerate(music_data['songs']):
            with col_song1 if idx % 2 == 0 else col_song2:
                st.markdown(f"""
                <div class='song-card'>
                    <p style='font-size: 1.3rem; font-weight: bold; margin: 0; color: {music_data['color']};'>
                        {idx + 1}. {song['title']}
                    </p>
                    <p style='font-size: 1rem; margin: 0.3rem 0 0 0; color: #b0bec5;'>
                        🎤 {song['artist']} • {song['year']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Section: Curated Playlists
        st.markdown("### 🎶 Curated Playlists")
        st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #b0bec5;'>Playlist yang dikurasi khusus untuk Anda</p>", unsafe_allow_html=True)
        
        col_playlist1, col_playlist2 = st.columns(2)
        
        for idx, playlist in enumerate(music_data['playlists']):
            with col_playlist1 if idx % 2 == 0 else col_playlist2:
                st.markdown(f"""
                <div class='playlist-card'>
                    <p style='font-size: 1.1rem; font-weight: 600; margin: 0;'>
                        {playlist}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Fun Facts Section
        st.markdown("### 💡 Did You Know?")
        
        fun_facts = {
            'rock': "🎸 Rock music meningkatkan motivasi dan energi hingga 40% saat berolahraga! Genre ini sempurna untuk workout dan aktivitas intens.",
            'paper': "🎹 Jazz dan Classical music dapat meningkatkan kreativitas hingga 50% dan membantu konsentrasi saat bekerja atau belajar.",
            'scissors': "⚡ Electronic music dengan beat teratur dapat meningkatkan fokus dan produktivitas. Perfect untuk coding atau creative work!"
        }
        
        st.markdown(f"""
        <div class='card' style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(138, 43, 226, 0.1));'>
            <p style='font-size: 1.1rem; text-align: center;'>{fun_facts.get(current_classification, 'Musik memiliki kekuatan luar biasa!')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mood Meter
        st.markdown("---")
        st.markdown("### 🎭 Mood Meter")
        
        mood_descriptions = {
            'rock': "Energetic, Powerful, Rebellious 🔥",
            'paper': "Calm, Sophisticated, Creative 🌊",
            'scissors': "Sharp, Modern, Futuristic ⚡"
        }
        
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem; background: rgba(30, 50, 60, 0.6); border-radius: 15px; border: 2px solid {music_data['color']};'>
            <p style='font-size: 1.5rem; font-weight: bold; color: {music_data['color']};'>
                Current Mood: {mood_descriptions.get(current_classification, 'Unknown')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class='for-you-alert'>
            <p>
                <span style='font-size: 2rem;'>🎵</span>
                <br>
                <span style='font-weight: bold;'>Lakukan Klasifikasi Gesture terlebih dahulu</span>
                <br>
                untuk mendapatkan rekomendasi musik yang dipersonalisasi untukmu!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎵 Preview Genres")
        st.markdown("<p style='text-align: center;'>Lihat preview genre musik yang tersedia</p>", unsafe_allow_html=True)
        
        col_preview1, col_preview2, col_preview3 = st.columns(3)
        
        with col_preview1:
            st.markdown("""
            <div class='menu-item'>
                <p style='font-size: 2.5rem;'>✊</p>
                <span style='font-weight: bold; color: #ff4444;'>Rock & Metal</span>
                <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Energetic & Powerful</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_preview2:
            st.markdown("""
            <div class='menu-item'>
                <p style='font-size: 2.5rem;'>✋</p>
                <span style='font-weight: bold; color: #44ff44;'>Jazz & Classical</span>
                <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Smooth & Sophisticated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_preview3:
            st.markdown("""
            <div class='menu-item'>
                <p style='font-size: 2.5rem;'>✌</p>
                <span style='font-weight: bold; color: #ffaa00;'>Electronic & Pop</span>
                <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Sharp & Modern</p>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TENTANG -----------------
with tabs[4]:
    clear_inactive_results(4)
    
    # Kontak Section
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Punya pertanyaan atau feedback? Hubungi tim kami!</p>
        <div style='margin-top: 1.5rem;'>
            <p><span style='font-weight: bold; color: #00d4ff;'>📍 Alamat:</span> Bandung, Banda Aceh Ujung</p>
            <p><span style='font-weight: bold; color: #00d4ff;'>📞 Telepon:</span> (021) 555-Saura</p>
            <p><span style='font-weight: bold; color: #00d4ff;'>📧 Email:</span> <a href='mailto:balqisisaura@gmail.com' style='color: #00d4ff !important; text-decoration: none;'>contact@aivision.ai</a></p>
            <p><span style='font-weight: bold; color: #00d4ff;'>🕒 Support:</span> 24/7 Online</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tentang Section
    st.markdown("<h2 class='section-title'>Tentang AI Vision ℹ</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.3rem;'>AI Vision adalah platform Computer Vision yang menggabungkan teknologi <span style='font-weight: bold; color: #00d4ff;'>Object Detection</span> dan <span style='font-weight: bold; color: #00d4ff;'>Image Classification</span>.</p>
        <p style='margin-top: 1rem;'>Teknologi yang digunakan:</p>
        <ul>
            <li><span style='font-weight: bold; color: #00d4ff;'>YOLO (You Only Look Once):</span> Untuk deteksi masker wajah secara real-time</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Deep Learning (H5 Model):</span> Untuk klasifikasi gesture tangan (Rock, Paper, Scissors)</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Streamlit:</span> Framework untuk antarmuka web yang interaktif</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Music Recommendation:</span> Sistem rekomendasi musik berdasarkan gesture</li>
        </ul>
        <p style='margin-top: 1.5rem;'>Dikembangkan oleh <span style='font-weight: bold; color: #00d4ff;'>Balqis Isaura</span></p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #00d4ff;'>#ComputerVision #AI #MachineLearning #MusicRecommendation</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2024 AI Vision. Dibuat dengan 🤖 dan ❤ oleh Balqis Isaura.</p>", unsafe_allow_html=True)
