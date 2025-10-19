import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
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
    st.header("🎯 Model PyTorch - YOLO")
    
    try:
        @st.cache_resource
        def load_yolo():
            return YOLO('model/Balqis Isaura_Laporan 4.pt')
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        
        st.success("✅ Model YOLO berhasil dimuat!")
        
        with st.sidebar.expander("📊 Info Model"):
            st.text(str(model.info()))
        
        st.markdown("### Upload Gambar untuk Deteksi Objek")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='yolo'
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔍 Deteksi Objek", type="primary"):
                with st.spinner("Mendeteksi objek..."):
                    results = model(image)
                    
                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        result_img = results[0].plot()
                        st.image(result_img, use_column_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Detail Deteksi")
                    
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        for i, box in enumerate(boxes, 1):
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.write(f"**{i}. {model.names[int(box.cls)]}**")
                            with col_b:
                                st.write(f"Confidence: **{box.conf[0]:.1%}**")
                    else:
                        st.info("ℹ️ Tidak ada objek terdeteksi")
                        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Pastikan file model ada di folder 'model/'")

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50")
    
    model = None
    
    try:
        @st.cache_resource
        def load_tensorflow():
            # Metode 1: Coba load dengan TensorFlow
            try:
                st.info("🔄 Mencoba load dengan TensorFlow...")
                return tf.keras.models.load_model(
                    'model/model_resnet50.h5',
                    compile=False
                )
            except Exception as e1:
                st.warning(f"⚠️ TensorFlow gagal: {str(e1)[:100]}")
                
                # Metode 2: Coba dengan safe_mode=False
                try:
                    st.info("🔄 Mencoba dengan safe_mode=False...")
                    return tf.keras.models.load_model(
                        'model/model_fixed.h5',
                        compile=False,
                        safe_mode=False
                    )
                except Exception as e2:
                    st.warning(f"⚠️ safe_mode=False gagal: {str(e2)[:100]}")
                    
                    # Metode 3: Coba dengan Keras langsung
                    try:
                        st.info("🔄 Mencoba dengan Keras...")
                        import keras
                        keras.config.disable_traceback_filtering()
                        return keras.models.load_model(
                            'model/model_fixed.h5',
                            compile=False
                        )
                    except Exception as e3:
                        st.error(f"❌ Semua metode gagal!")
                        raise e3
        
        with st.spinner("Loading TensorFlow model..."):
            model = load_tensorflow()
        
        st.success("✅ Model TensorFlow berhasil dimuat!")
        
        with st.sidebar.expander("📊 Architecture Model"):
            try:
                from io import StringIO
                stream = StringIO()
                model.summary(print_fn=lambda x: stream.write(x + '\n'))
                st.text(stream.getvalue())
            except:
                st.text("Model summary tidak tersedia")
        
        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='tf'
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)
            
            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    try:
                        # Preprocess image
                        img_array = np.array(image.resize((224, 224)))
                        
                        # Convert RGBA to RGB jika perlu
                        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
                            img_array = img_array[:, :, :3]
                        
                        # Pastikan RGB (3 channels)
                        if len(img_array.shape) == 2:  # Grayscale
                            img_array = np.stack([img_array] * 3, axis=-1)
                        
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                        
                        # Prediksi
                        predictions = model.predict(img_array, verbose=0)
                        
                        with col2:
                            st.subheader("🎯 Hasil Prediksi")
                            
                            predicted_class = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class]
                            
                            st.metric("Kelas Prediksi", f"Class {predicted_class}", 
                                     help="Index kelas yang diprediksi")
                            st.metric("Confidence", f"{confidence:.2%}",
                                     help="Tingkat kepercayaan model")
                            
                            with st.expander("📊 Lihat Semua Probabilitas"):
                                for i, prob in enumerate(predictions[0]):
                                    st.progress(float(prob), text=f"Class {i}: {prob:.4f}")
                    
                    except Exception as pred_error:
                        st.error(f"❌ Error saat prediksi: {pred_error}")
        
    except Exception as e:
        st.error(f"❌ Model tidak bisa dimuat: {str(e)}")
        st.info("""
        **💡 Solusi:**
        1. Model mungkin disimpan dengan versi TensorFlow/Keras yang berbeda
        2. Coba convert ulang model dengan script berikut di laptop:
        
        ```python
        import tensorflow as tf
        model = tf.keras.models.load_model('model_fixed.h5')
        model.save('model_fixed_new.h5')
        ```
        
        3. Atau gunakan model YOLO yang sudah berfungsi dengan baik
        """)

st.markdown("---")
st.markdown("**📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")
