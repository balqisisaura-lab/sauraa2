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

# ========================== CUSTOM CSS ==========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stApp {
    background: radial-gradient(circle at 10% 20%, #001f3f, #000814);
    color: #FFFFFF;
}
h1, h2, h3 {
    color: #00d4ff;
    text-align: center;
    font-weight: 600;
}
.stButton > button {
    background: linear-gradient(90deg, #00d4ff, #0091ea);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0091ea, #00d4ff);
    box-shadow: 0px 0px 15px #00d4ff;
    transform: scale(1.05);
}
.success-blink {
    font-size: 1.5rem;
    font-weight: bold;
    color: #00ff99;
    border: 2px solid #00ff99;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    animation: blink 1s infinite;
}
@keyframes blink {
    50% {opacity: 0.5;}
}
footer {
    text-align: center;
    color: #ccc;
    font-size: 0.9em;
    margin-top: 2rem;
}

/* === GAME RPS STYLE === */
.rps-container {
    text-align: center;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ========================== MODEL LOADING ==========================
mask_model = YOLO("yolov8n.pt")  # contoh model YOLO
gesture_model = YOLO("yolov8n.pt")  # contoh model YOLO gesture

# ========================== APP TITLE ==========================
st.title("🤖 AI Vision - Mask Detection & Hand Gesture")

# ========================== SIDEBAR ==========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.markdown("## Navigasi")
st.sidebar.info("Gunakan menu di bawah untuk berpindah fitur.")
tabs = st.tabs(["🏠 Beranda", "😷 Deteksi Masker", "✊✋✌ Klasifikasi Gesture", "🎮 Game RPS", "🎯 Keahlian Mu", "📞 Kontak", "ℹ Tentang"])

# ========================== TAB 1: HOME ==========================
with tabs[0]:
    st.header("Selamat Datang di AI Vision! 🌟")
    st.write("Aplikasi ini menggunakan teknologi **Computer Vision** untuk mendeteksi masker dan mengenali gesture tangan secara real-time.")
    st.image("https://cdn.dribbble.com/users/1162077/screenshots/3848914/programmer.gif", use_container_width=True)

# ========================== TAB 2: MASK DETECTION ==========================
with tabs[1]:
    st.header("😷 Deteksi Masker")
    uploaded_file = st.file_uploader("Unggah gambar wajah:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar Diupload', use_container_width=True)
        results = mask_model.predict(image)
        st.success("✅ Deteksi masker selesai!")
        st.image(image, caption="Hasil deteksi", use_container_width=True)

# ========================== TAB 3: HAND GESTURE ==========================
with tabs[2]:
    st.header("✊✋✌ Klasifikasi Gesture Tangan")
    uploaded_gesture = st.file_uploader("Unggah gambar tangan:", type=["jpg", "jpeg", "png"])
    if uploaded_gesture:
        image_gesture = Image.open(uploaded_gesture)
        st.image(image_gesture, caption='Gambar Gesture Diupload', use_container_width=True)
        results_gesture = gesture_model.predict(image_gesture)
        st.success("🤝 Klasifikasi gesture selesai!")

# ========================== TAB 4: GAME RPS ==========================
with tabs[3]:
    st.header("🎮 Rock Paper Scissors Game")
    st.markdown("<div class='rps-container'>", unsafe_allow_html=True)

    if 'score_user' not in st.session_state:
        st.session_state['score_user'] = 0
        st.session_state['score_ai'] = 0
        st.session_state['user_choice'] = None
        st.session_state['ai_choice'] = None
        st.session_state['result'] = None

    def reset_game():
        st.session_state['user_choice'] = None
        st.session_state['ai_choice'] = None
        st.session_state['result'] = None
        st.session_state['score_user'] = 0
        st.session_state['score_ai'] = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✊ Rock", key="rock_btn"):
            st.session_state['user_choice'] = "Rock"
    with col2:
        if st.button("✋ Paper", key="paper_btn"):
            st.session_state['user_choice'] = "Paper"
    with col3:
        if st.button("✌ Scissors", key="scissors_btn"):
            st.session_state['user_choice'] = "Scissors"

    if st.session_state['user_choice']:
        st.session_state['ai_choice'] = np.random.choice(["Rock", "Paper", "Scissors"])
        user = st.session_state['user_choice']
        ai = st.session_state['ai_choice']

        if user == ai:
            st.session_state['result'] = "Draw"
        elif (user == "Rock" and ai == "Scissors") or \
             (user == "Scissors" and ai == "Paper") or \
             (user == "Paper" and ai == "Rock"):
            st.session_state['result'] = "You Win"
            st.session_state['score_user'] += 1
        else:
            st.session_state['result'] = "AI Wins"
            st.session_state['score_ai'] += 1

    st.subheader("🧠 Hasil Pertandingan")
    st.write(f"Pilihan Kamu: {st.session_state['user_choice']}")
    st.write(f"Pilihan AI: {st.session_state['ai_choice']}")
    if st.session_state['result'] == "You Win":
        st.markdown("<div class='success-blink'>🏆 KAMU MENANG!</div>", unsafe_allow_html=True)
    elif st.session_state['result'] == "AI Wins":
        st.markdown("<div class='success-blink' style='border-color:#ff4d4d; color:#ff4d4d;'>💀 KALAH!</div>", unsafe_allow_html=True)
    elif st.session_state['result'] == "Draw":
        st.info("🤝 SERI!")

    st.markdown("---")
    st.subheader("📊 Skor")
    st.write(f"👤 Kamu: {st.session_state['score_user']} | 🤖 AI: {st.session_state['score_ai']}")
    if st.button("🔄 Reset Game", key="reset_btn"):
        reset_game()
        st.success("Permainan direset!")

# ========================== TAB 5: KEAHLIAN ==========================
with tabs[4]:
    st.header("🎯 Keahlian Mu")
    st.write("""
    - Deteksi masker wajah dengan YOLOv8  
    - Pengenalan gesture tangan  
    - Game interaktif AI (Rock Paper Scissors)
    """)

# ========================== TAB 6: KONTAK ==========================
with tabs[5]:
    st.header("📞 Hubungi Kami")
    st.markdown("""
    <p><span style='font-weight: bold; color: #00d4ff;'>📍 Alamat:</span> Jl. AI Technology No. 42, Tech City</p>
    <p><span style='font-weight: bold; color: #00d4ff;'>📞 Telepon:</span> (021) 555-VISION</p>
    <p><span style='font-weight: bold; color: #00d4ff;'>📧 Email:</span> support@aivision.tech</p>
    """, unsafe_allow_html=True)

# ========================== TAB 7: TENTANG ==========================
with tabs[6]:
    st.header("ℹ Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan sebagai proyek **AI Vision System** yang menggabungkan teknologi deteksi objek, 
    pengenalan gesture tangan, dan permainan interaktif berbasis kecerdasan buatan.  
    Dibangun menggunakan **Streamlit**, **TensorFlow**, dan **YOLOv8**.
    """)

# ========================== FOOTER ==========================
st.markdown("<footer>© 2025 AI Vision | Created with ❤️ using Streamlit</footer>", unsafe_allow_html=True)
