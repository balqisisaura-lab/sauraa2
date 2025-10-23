import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os 
from io import BytesIO

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="AI Vision - Mask Detection & Hand Gesture",
    page_icon="🤖",
    layout="wide"
)

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
        
        .recommendation-alert {
            padding: 1.5rem;
            border: 3px solid #ffd700; 
            border-radius: 15px;
            background: linear-gradient(45deg, rgba(255, 215, 0, 0.1), rgba(255, 193, 7, 0.1)); 
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
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
        
        /* Animasi Glowing Orbs */
        .orbs-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
            overflow: hidden;
        }
        
        .orb {
            position: absolute;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(0, 212, 255, 0.8), rgba(0, 145, 234, 0.4));
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.8), 0 0 40px rgba(0, 212, 255, 0.5);
            animation: fall linear forwards;
            opacity: 0;
        }
        
        @keyframes fall {
            0% {
                opacity: 0;
                transform: translateY(-100px) scale(0);
            }
            10% {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
            90% {
                opacity: 1;
            }
            100% {
                opacity: 0;
                transform: translateY(100vh) scale(0.5);
            }
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

# Game State
if 'game_score_win' not in st.session_state:
    st.session_state['game_score_win'] = 0
if 'game_score_lose' not in st.session_state:
    st.session_state['game_score_lose'] = 0
if 'game_score_draw' not in st.session_state:
    st.session_state['game_score_draw'] = 0
if 'game_player_gesture' not in st.session_state:
    st.session_state['game_player_gesture'] = None
if 'game_ai_gesture' not in st.session_state:
    st.session_state['game_ai_gesture'] = None
if 'game_result' not in st.session_state:
    st.session_state['game_result'] = None
if 'last_game_uploader' not in st.session_state:
    st.session_state['last_game_uploader'] = None
if 'game_player_image' not in st.session_state:
    st.session_state['game_player_image'] = None

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
    
    if current_tab_index != 3:
        if st.session_state.get('game_player_gesture') is not None:
            st.session_state['game_player_gesture'] = None
        if st.session_state.get('game_ai_gesture') is not None:
            st.session_state['game_ai_gesture'] = None
        if st.session_state.get('game_result') is not None:
            st.session_state['game_result'] = None

# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["🏠 Beranda", "😷 Deteksi Masker", "✊✋✌ Klasifikasi Gesture", "🎮 Game RPS", "🎯 Rekomendasi", "📞 Kontak", "ℹ Tentang"])

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
            <p style='font-size: 3rem;'>🎯</p>
            <span style='font-weight: bold; color: #00d4ff;'>Rekomendasi Cerdas</span>
            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Dapatkan saran berdasarkan hasil klasifikasi gesture Anda.</p>
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
                            
                            # Animasi Glowing Orbs
                            st.markdown("""
                            <div class='orbs-container' id='orbs-detection'></div>
                            <script>
                                const container = document.getElementById('orbs-detection');
                                for(let i = 0; i < 25; i++) {
                                    const orb = document.createElement('div');
                                    orb.className = 'orb';
                                    orb.style.left = Math.random() * 100 + '%';
                                    orb.style.width = orb.style.height = (Math.random() * 15 + 10) + 'px';
                                    orb.style.animationDuration = (Math.random() * 2 + 2) + 's';
                                    orb.style.animationDelay = (Math.random() * 0.5) + 's';
                                    container.appendChild(orb);
                                }
                                setTimeout(() => container.remove(), 4000);
                            </script>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<div class='success-blink'>✅ DETEKSI BERHASIL!</div>", unsafe_allow_html=True)
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
                image_rgb = image_pil.convert('RGB')
                image_class_resized = image_rgb.resize((128, 128))
                
                st.session_state['classification_image_input'] = image_class_resized
                st.image(st.session_state['classification_image_input'], caption="Gambar Input (128x128 RGB)", use_container_width=True)

                if st.button("🎯 Klasifikasikan Gesture", type="primary", key="classify_btn"):
                    with st.spinner("⏳ Mengklasifikasikan gesture dengan AI..."):
                        try:
                            img_array = np.array(image_class_resized)
                            img_array = img_array / 255.0
                            preprocessed_img = np.expand_dims(img_array, axis=0)
                            predictions = classification_model.predict(preprocessed_img)
                            class_names = ['Rock', 'Paper', 'Scissors']
                            predicted_class_idx = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class_idx] * 100
                            final_result = class_names[predicted_class_idx]
                            
                            st.session_state['classification_final_result'] = final_result
                            st.session_state['classification'] = final_result.lower()
                            
                            st.markdown("""
                            <div class='orbs-container' id='orbs-classification'></div>
                            <script>
                                const container = document.getElementById('orbs-classification');
                                for(let i = 0; i < 25; i++) {
                                    const orb = document.createElement('div');
                                    orb.className = 'orb';
                                    orb.style.left = Math.random() * 100 + '%';
                                    orb.style.width = orb.style.height = (Math.random() * 15 + 10) + 'px';
                                    orb.style.animationDuration = (Math.random() * 2 + 2) + 's';
                                    orb.style.animationDelay = (Math.random() * 0.5) + 's';
                                    container.appendChild(orb);
                                }
                                setTimeout(() => container.remove(), 4000);
                            </script>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<div class='success-blink'>✨ KLASIFIKASI BERHASIL!</div>", unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan: {e}")

        with col_class_output:
            if st.session_state.get('classification_final_result') is not None:
                final_result = st.session_state['classification_final_result']
                
                st.markdown("### 🎯 Hasil Klasifikasi AI")
                
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

# ----------------- GAME ROCK PAPER SCISSORS (FIXED) -----------------
with tabs[3]:
    clear_inactive_results(3)
    st.markdown("<h2 class='section-title'>🎮 Rock Paper Scissors Game</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='text-align: center;'>Tantang AI dalam permainan <span style='font-weight: bold; color: #00d4ff;'>Batu-Gunting-Kertas</span>! Upload gesture kamu dan lihat siapa yang menang!</p>
    </div>
    """, unsafe_allow_html=True)
    
    H5_MODEL_PATH = 'model/compressed.h5'
    game_model = load_classification_model(H5_MODEL_PATH)
    
    if game_model:
        # Score Board
        col_score1, col_score2, col_score3 = st.columns(3)
        with col_score1:
            st.markdown(f"""
            <div class='menu-item' style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 200, 100, 0.2)); border-color: #00ff88;'>
                <p style='font-size: 2rem; margin: 0;'>🏆</p>
                <p style='font-size: 1.5rem; margin: 0; color: #00ff88 !important; font-weight: bold;'>{st.session_state['game_score_win']}</p>
                <p style='margin: 0;'>MENANG</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_score2:
            st.markdown(f"""
            <div class='menu-item' style='background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 193, 7, 0.2)); border-color: #ffd700;'>
                <p style='font-size: 2rem; margin: 0;'>🤝</p>
                <p style='font-size: 1.5rem; margin: 0; color: #ffd700 !important; font-weight: bold;'>{st.session_state['game_score_draw']}</p>
                <p style='margin: 0;'>SERI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_score3:
            st.markdown(f"""
            <div class='menu-item' style='background: linear-gradient(135deg, rgba(255, 82, 82, 0.2), rgba(255, 23, 68, 0.2)); border-color: #ff5252;'>
                <p style='font-size: 2rem; margin: 0;'>💔</p>
                <p style='font-size: 1.5rem; margin: 0; color: #ff5252 !important; font-weight: bold;'>{st.session_state['game_score_lose']}</p>
                <p style='margin: 0;'>KALAH</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Upload Section
        col_upload_center = st.columns([1, 2, 1])[1]
        with col_upload_center:
            uploaded_game_file = st.file_uploader("📸 Upload Gesture Kamu", type=["jpg", "jpeg", "png"], key="game_uploader")
            
            if st.session_state.get('last_game_uploader') != uploaded_game_file:
                st.session_state['game_player_gesture'] = None
                st.session_state['game_ai_gesture'] = None
                st.session_state['game_result'] = None
                st.session_state['game_player_image'] = None
                st.session_state['last_game_uploader'] = uploaded_game_file
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Game Area
        col_game_input, col_game_vs, col_game_ai = st.columns([2, 1, 2])
        
        with col_game_input:
            st.markdown("### 👤 PLAYER")
            
            if uploaded_game_file:
                image_pil = Image.open(uploaded_game_file)
                image_rgb = image_pil.convert('RGB')
                image_resized = image_rgb.resize((128, 128))
                
                # Simpan image untuk ditampilkan nanti
                st.session_state['game_player_image'] = image_resized
                
                # Tampilkan preview image
                st.image(image_resized, caption="Gesture Kamu", use_container_width=True)
                
                # Tampilkan hasil klasifikasi jika sudah ada
                if st.session_state.get('game_player_gesture'):
                    gesture_icons = {'Rock': '✊', 'Paper': '✋', 'Scissors': '✌'}
                    player_gesture = st.session_state['game_player_gesture']
                    icon = gesture_icons.get(player_gesture, '🤖')
                    
                    st.markdown(f"""
                    <div class='card' style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 200, 100, 0.2)); margin-top: 1rem; padding: 1.5rem;'>
                        <p style='font-size: 4rem; margin: 0; text-align: center;'>{icon}</p>
                        <p style='font-size: 1.8rem; margin: 0.5rem 0 0 0; text-align: center; color: #00ff88 !important; font-weight: bold;'>{player_gesture}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tombol Main
                if not st.session_state.get('game_player_gesture'):
                    if st.button("⚔️ MAIN!", type="primary", key="play_game_btn", use_container_width=True):
                        with st.spinner("🎲 AI sedang berpikir..."):
                            try:
                                # Prediksi gesture player
                                img_array = np.array(image_resized) / 255.0
                                preprocessed_img = np.expand_dims(img_array, axis=0)
                                predictions = game_model.predict(preprocessed_img)
                                class_names = ['Rock', 'Paper', 'Scissors']
                                predicted_class_idx = np.argmax(predictions[0])
                                player_gesture = class_names[predicted_class_idx]
                                
                                # AI random pilih gesture
                                ai_gesture = np.random.choice(class_names)
                                
                                # Tentukan pemenang
                                if player_gesture == ai_gesture:
                                    result = "SERI 🤝"
                                    st.session_state['game_score_draw'] += 1
                                    result_color = "#ffd700"
                                    result_icon = "🤝"
                                elif (player_gesture == "Rock" and ai_gesture == "Scissors") or \
                                     (player_gesture == "Paper" and ai_gesture == "Rock") or \
                                     (player_gesture == "Scissors" and ai_gesture == "Paper"):
                                    result = "KAMU MENANG! 🎉"
                                    st.session_state['game_score_win'] += 1
                                    result_color = "#00ff88"
                                    result_icon = "🎉"
                                else:
                                    result = "AI MENANG! 🤖"
                                    st.session_state['game_score_lose'] += 1
                                    result_color = "#ff5252"
                                    result_icon = "😢"
                                
                                # Simpan hasil
                                st.session_state['game_player_gesture'] = player_gesture
                                st.session_state['game_ai_gesture'] = ai_gesture
                                st.session_state['game_result'] = {
                                    'text': result,
                                    'color': result_color,
                                    'icon': result_icon
                                }
                                
                                # Animasi Glowing Orbs
                                st.markdown("""
                                <div class='orbs-container' id='orbs-game'></div>
                                <script>
                                    const container = document.getElementById('orbs-game');
                                    for(let i = 0; i < 30; i++) {
                                        const orb = document.createElement('div');
                                        orb.className = 'orb';
                                        orb.style.left = Math.random() * 100 + '%';
                                        orb.style.width = orb.style.height = (Math.random() * 15 + 10) + 'px';
                                        orb.style.animationDuration = (Math.random() * 2 + 2) + 's';
                                        orb.style.animationDelay = (Math.random() * 0.5) + 's';
                                        container.appendChild(orb);
                                    }
                                    setTimeout(() => container.remove(), 4000);
                                </script>
                                """, unsafe_allow_html=True)
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Terjadi kesalahan: {e}")
            else:
                st.markdown("""
                <div class='card' style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 145, 234, 0.1)); min-height: 200px; display: flex; align-items: center; justify-content: center; flex-direction: column;'>
                    <p style='font-size: 3rem; margin: 0;'>📸</p>
                    <p style='font-size: 1.2rem; margin: 1rem 0 0 0; text-align: center;'>Upload gesture untuk mulai bermain</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_game_vs:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if st.session_state.get('game_result'):
                result_data = st.session_state['game_result']
                st.markdown(f"""
                <div style='text-align: center; margin-top: 2rem;'>
                    <p style='font-size: 4rem; margin: 0;'>{result_data['icon']}</p>
                    <p style='font-size: 1.3rem; margin: 0.5rem 0 0 0; font-weight: bold; color: {result_data['color']} !important;'>{result_data['text']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; margin-top: 2rem;'>
                    <p style='font-size: 4rem; margin: 0;'>⚔️</p>
                    <p style='font-size: 1.5rem; margin: 0.5rem 0 0 0; color: #00d4ff !important; font-weight: bold;'>VS</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_game_ai:
            st.markdown("### 🤖 AI")
            if st.session_state.get('game_ai_gesture'):
                gesture_icons = {'Rock': '✊', 'Paper': '✋', 'Scissors': '✌'}
                ai_gesture = st.session_state['game_ai_gesture']
                icon = gesture_icons.get(ai_gesture, '🤖')
                
                st.markdown(f"""
                <div class='card' style='background: linear-gradient(135deg, rgba(255, 82, 82, 0.2), rgba(255, 23, 68, 0.2)); border: 2px solid rgba(255, 82, 82, 0.5); min-height: 280px; display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2rem;'>
                    <p style='font-size: 6rem; margin: 0; line-height: 1;'>{icon}</p>
                    <p style='font-size: 2rem; margin: 1.5rem 0 0 0; color: #ff5252 !important; font-weight: bold; text-transform: uppercase;'>{ai_gesture}</p>
                    <p style='font-size: 1rem; margin: 0.5rem 0 0 0; color: #ff8888 !important;'>AI Choice</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='card' style='background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 145, 234, 0.1)); border: 2px solid rgba(0, 212, 255, 0.3); min-height: 280px; display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2rem;'>
                    <p style='font-size: 5rem; margin: 0; line-height: 1;'>❓</p>
                    <p style='font-size: 1.4rem; margin: 1.5rem 0 0 0; color: #00d4ff !important; font-weight: 600;'>Waiting...</p>
                    <p style='font-size: 1rem; margin: 0.5rem 0 0 0; color: #78909c !important;'>Upload gesture & klik MAIN!</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action Buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            if st.session_state.get('game_player_gesture'):
                if st.button("🔄 Main Lagi", type="primary", key="play_again", use_container_width=True):
                    st.session_state['game_player_gesture'] = None
                    st.session_state['game_ai_gesture'] = None
                    st.session_state['game_result'] = None
                    st.session_state['last_game_uploader'] = None
                    st.rerun()
        
        with col_btn3:
            if st.button("🔄 Reset Skor", type="secondary", key="reset_score", use_container_width=True):
                st.session_state['game_score_win'] = 0
                st.session_state['game_score_lose'] = 0
                st.session_state['game_score_draw'] = 0
                st.session_state['game_player_gesture'] = None
                st.session_state['game_ai_gesture'] = None
                st.session_state['game_result'] = None
                st.rerun()
    
    else:
        st.warning(f"⚠ Model tidak dapat dimuat dari '{H5_MODEL_PATH}'.")

# ----------------- REKOMENDASI -----------------
with tabs[4]:
    clear_inactive_results(4)
    st.markdown("<h2 class='section-title'>Rekomendasi Berdasarkan Gesture 🎯</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rekomendasi ini didasarkan pada hasil klasifikasi gesture Anda.</p>", unsafe_allow_html=True)
    
    recommendations = {
        'rock': {
            'title': '✊ ROCK - Kekuatan & Ketegasan',
            'items': [
                {'nama': 'Action Games', 'deskripsi': 'Game penuh aksi dan tantangan yang membutuhkan ketegasan.', 'kategori': 'Gaming'},
                {'nama': 'Strength Training', 'deskripsi': 'Program latihan kekuatan untuk membangun otot.', 'kategori': 'Fitness'}
            ]
        },
        'paper': {
            'title': '✋ PAPER - Fleksibilitas & Kreativitas',
            'items': [
                {'nama': 'Creative Writing', 'deskripsi': 'Kursus menulis kreatif untuk mengembangkan imajinasi.', 'kategori': 'Education'},
                {'nama': 'Digital Art', 'deskripsi': 'Belajar seni digital dan desain grafis.', 'kategori': 'Art'}
            ]
        },
        'scissors': {
            'title': '✌ SCISSORS - Ketepatan & Strategi',
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
        <div class='card' style='background: linear-gradient(45deg, rgba(0, 212, 255, 0.2), rgba(0, 145, 234, 0.2)); border-color: #00d4ff;'>
            <p style='font-size: 1.8rem; text-align: center; color: #00d4ff; font-weight: bold;'>{rec_data['title']}</p>
            <p style='font-size: 1.1rem; text-align: center;'>Berdasarkan gesture Anda, kami merekomendasikan:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🎮 Rekomendasi Utama")
            for item in rec_data['items']:
                st.markdown(f"""
                <div class='menu-item'>
                    <span style='font-weight: bold; color: #00d4ff;'>{item['nama']}</span>
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
with tabs[5]:
    clear_inactive_results(5)
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Punya pertanyaan atau feedback? Hubungi tim kami!</p>
        <div style='margin-top: 1.5rem;'>
            <p><span style='font-weight: bold; color: #00d4ff;'>📍 Alamat:</span> USA</p>
            <p><span style='font-weight: bold; color: #00d4ff;'>📞 Telepon:</span> (021) 555-VISION</p>
            <p><span style='font-weight: bold; color: #00d4ff;'>📧 Email:</span> <a href='mailto:contact@aivision.ai' style='color: #00d4ff !important; text-decoration: none;'>contact@aivision.ai</a></p>
            <p><span style='font-weight: bold; color: #00d4ff;'>🕒 Support:</span> 24/7 Online tapi jangan spam</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- TENTANG -----------------
with tabs[6]:
    clear_inactive_results(6)
    st.markdown("<h2 class='section-title'>Tentang AI Vision ℹ</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.3rem;'>AI Vision adalah platform Computer Vision yang menggabungkan teknologi <span style='font-weight: bold; color: #00d4ff;'>Object Detection</span> dan <span style='font-weight: bold; color: #00d4ff;'>Image Classification</span>.</p>
        <p style='margin-top: 1rem;'>Teknologi yang digunakan:</p>
        <ul>
            <li><span style='font-weight: bold; color: #00d4ff;'>YOLO (You Only Look Once):</span> Untuk deteksi masker wajah secara real-time</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Deep Learning (H5 Model):</span> Untuk klasifikasi gesture tangan (Rock, Paper, Scissors)</li>
            <li><span style='font-weight: bold; color: #00d4ff;'>Streamlit:</span> Framework untuk antarmuka web yang interaktif</li>
        </ul>
        <p style='margin-top: 1.5rem;'>Dikembangkan oleh <span style='font-weight: bold; color: #00d4ff;'>Balqis Isaura</span></p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #00d4ff;'>#ComputerVision #AI #MachineLearning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2025 AI Vision. Dibuat dengan 🤖 dan ❤ oleh Balqis Isaura.</p>", unsafe_allow_html=True)
