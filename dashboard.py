import streamlit as st

# Test sederhana dulu
st.set_page_config(
    page_title="AI Vision Test",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
    }
    
    .main-title {
        text-align: center;
        font-size: 4rem;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>AI VISION TEST</h1>", unsafe_allow_html=True)
st.write("Jika teks ini muncul, berarti Streamlit berjalan dengan baik!")

# Test import libraries
try:
    from ultralytics import YOLO
    st.success("✅ YOLO berhasil di-import")
except Exception as e:
    st.error(f"❌ YOLO error: {e}")

try:
    import tensorflow as tf
    st.success("✅ TensorFlow berhasil di-import")
except Exception as e:
    st.error(f"❌ TensorFlow error: {e}")

try:
    from PIL import Image
    st.success("✅ PIL berhasil di-import")
except Exception as e:
    st.error(f"❌ PIL error: {e}")

try:
    import numpy as np
    st.success("✅ NumPy berhasil di-import")
except Exception as e:
    st.error(f"❌ NumPy error: {e}")
