import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
import os
from io import StringIO

# ==================== CONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Dashboard Deteksi - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# ==================== MODEL PYTORCH YOLO ====================
if model_choice == "PyTorch - YOLO":
    st.header("🔥 Model PyTorch - YOLO")

    try:
        # Tanpa cache karena error unpickling di PyTorch >= 2.6
        def load_yolo():
            # Izinkan global class untuk safe load (fix UnpicklingError)
            torch.serialization.add_safe_globals([__import__('ultralytics').nn.tasks.DetectionModel])
            
            model_path = "model/Balqis Isaura_Laporan 4.pt"
            if not os.path.exists(model_path):
                st.error("❌ File model YOLO tidak ditemukan.")
                st.stop()
            return YOLO(model_path)
        
        with st.spinner("🔄 Memuat model YOLO..."):
            model = load_yolo()

        st.success("✅ Model YOLO berhasil dimuat!")

        # Info model di sidebar
        with st.sidebar.expander("📊 Info Model YOLO"):
            info_text = StringIO()
            model.info(verbose=True)
            st.text("Model siap digunakan untuk deteksi objek.")

        # Upload gambar
        st.markdown("### Upload Gambar untuk Deteksi Objek")
        uploaded_file = st.file_uploader(
            "Pilih gambar (jpg/jpeg/png)",
            type=["jpg", "jpeg", "png"],
            key="yolo"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔍 Jalankan Deteksi YOLO", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    results = model.predict(source=np.array(image), conf=0.5, verbose=False)
                    result_img = results[0].plot()

                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        st.image(result_img, use_column_width=True)

                    st.subheader("📋 Detail Deteksi")
                    boxes = results[0].boxes

                    if len(boxes) > 0:
                        for i, box in enumerate(boxes, 1):
                            label = results[0].names[int(box.cls)]
                            conf = float(box.conf)
                            st.write(f"- **{i}. {label}** ({conf:.2%})")
                    else:
                        st.warning("⚠️ Tidak ada objek terdeteksi.")

    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO.\n**Error:** {e}")
        st.info("""
        **Solusi yang bisa dicoba:**
        1. Pastikan file `Balqis Isaura_Laporan 4.pt` kompatibel dengan PyTorch 2.6+
        2. Re-export model di laptop kamu dengan perintah:
           ```python
           from ultralytics import YOLO
           m = YOLO('Balqis Isaura_Laporan 4.pt')
           m.export(format='pt')
           ```
        """)

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50")

    try:
        @st.cache_resource
        def load_resnet():
            model_path = "model/model_resnet50.keras"
            if not os.path.exists(model_path):
                st.error("❌ File model TensorFlow tidak ditemukan.")
                st.stop()
            return tf.keras.models.load_model(model_path, compile=False)

        with st.spinner("🔄 Memuat model ResNet50..."):
            model = load_resnet()

        st.success("✅ Model TensorFlow berhasil dimuat!")

        # Sidebar info model
        with st.sidebar.expander("📊 Info Model ResNet"):
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + "\n"))
            st.text(stream.getvalue())

        # Upload gambar
        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar (jpg/jpeg/png)",
            type=["jpg", "jpeg", "png"],
            key="tf"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔮 Jalankan Prediksi", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    img_array = np.array(image.resize((224, 224)))

                    # Handle grayscale atau RGBA
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                    preds = model.predict(img_array, verbose=0)
                    pred_class = np.argmax(preds[0])
                    confidence = preds[0][pred_class]

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Kelas Prediksi", f"Class {pred_class}")
                        st.metric("Confidence", f"{confidence:.2%}")

                        with st.expander("📊 Probabilitas Lengkap"):
                            for i, p in enumerate(preds[0]):
                                st.progress(float(p), text=f"Class {i}: {p:.4f}")

    except Exception as e:
        st.error(f"❌ Error memuat model TensorFlow:\n{e}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("**📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")
