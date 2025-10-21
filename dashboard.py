import streamlit as st

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

# ===================== STYLE SESUAI DESAIN GAMBAR =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

body, [data-testid="stAppViewContainer"] {
    font-family: 'Poppins', sans-serif;
    background-color: #fffdfd;
}

#MainMenu, header, footer {visibility: hidden;}

.header-container {
    text-align: center;
    padding-top: 50px;
    padding-bottom: 50px;
    position: relative;
}

.header-badge {
    display: inline-block;
    border: 1.5px solid #9D8CFF;
    color: #7C4DFF;
    border-radius: 25px;
    padding: 6px 22px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}

.header-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin: 15px 0 5px;
    color: #00C8FF;
}

.header-subtitle {
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7C4DFF, #00C8FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.button-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 30px;
}

.option-button {
    border: 1.5px solid #9D8CFF;
    background: white;
    color: #7C4DFF;
    border-radius: 25px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.option-button:hover {
    background: linear-gradient(90deg, #7C4DFF, #00C8FF);
    color: white;
}

.grid-bottom {
    margin-top: 70px;
    height: 120px;
    background: linear-gradient(90deg, #00C8FF, #7C4DFF);
    position: relative;
    overflow: hidden;
}

.grid-bottom::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px);
    background-size: 50px 50px;
    transform: perspective(400px) rotateX(60deg);
    transform-origin: bottom;
}

</style>
""", unsafe_allow_html=True)

# ===================== KONTEN HALAMAN =====================
st.markdown("""
<div class="header-container">
    <div class="header-badge">BY BALQIS ISAURA</div>
    <div class="header-title">DASHBOARD</div>
    <div class="header-subtitle">DETEKSI OBJEK & KLASIFIKASI GAMBAR</div>
    <div class="button-container">
        <button class="option-button">DETEKSI OBJEK</button>
        <button class="option-button">KLASIFIKASI GAMBAR</button>
    </div>
</div>

<div class="grid-bottom"></div>
""", unsafe_allow_html=True)
