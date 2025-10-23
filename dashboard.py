import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os 
from io import BytesIO

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Pijjahut - Dini Arifatul Nasywa",
    page_icon="🍕",
    layout="wide"
)

# ========================== CUSTOM STYLE (Fokus pada Transisi Halus) ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Dancing+Script&family=Great+Vibes&display=swap');

        /* === Efek Transisi Fade-In Global untuk Tampilan Tab yang Lebih Halus === */
        .stApp {
            animation: fadeInAnimation ease 0.4s; 
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }

        @keyframes fadeInAnimation {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        /* ====================================================================== */

        body, .stApp {
            background: linear-gradient(135deg, #fff8dc 0%, #f5f5dc 50%, #ede0c8 100%);
            color: #3e2723;
            font-family: 'Arial', sans-serif;
            overflow-x: hidden;
        }

        * {
            color: #3e2723 !important;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); 
        }

        .main-title {
            text-align: center;
            font-size: 15vw; 
            font-weight: 900;
            font-family: 'Great Vibes', cursive;
            color: #cc0000 !important;
            text-shadow: 6px 6px 20px rgba(0,0,0,0.6), 0 0 50px rgba(204,0,0,0.8); 
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            animation: bounceIn 2s ease-in-out, pulse 3s infinite;
            position: relative;
            width: 100%;
            display: block; 
            letter-spacing: 15px; 
        }
        
        .subtitle {
            text-align: center;
            color: #5d4037 !important;
            font-size: 1.8rem;
            margin-bottom: 2rem;
            font-style: italic;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.7);
            animation: fadeInUp 2s ease-in-out;
        }
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ff5722 !important;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            margin-top: 2rem;
            text-align: center;
            font-family: 'Pacifico', cursive;
            animation: slideInLeft 1.5s ease-in-out;
        }
        .stButton > button {
            background: linear-gradient(45deg, #ff5722, #e64a19);
            color: #fff !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            padding: 1rem 2rem !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            transition: all 0.4s ease;
            font-size: 1.1rem;
            border: none;
            position: relative;
            overflow: hidden;
        }
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            border: 3px solid #d7ccc8;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .menu-item {
            background: linear-gradient(45deg, rgba(255, 248, 220, 0.95), rgba(255, 235, 204, 0.95));
            border-radius: 20px;
            padding: 2rem;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            font-size: 1.1rem;
            position: relative;
        }
        
        /* CUSTOM CSS FILE UPLOADER */
        .stFileUploader {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .stFileUploader > div:first-child > div:first-child {
            background-color: rgba(255, 255, 255, 0.9);
            border: 3px dashed #ff5722; 
            border-radius: 25px; 
            padding: 2.5rem 1.5rem; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
            transition: all 0.3s ease;
        }
        .stFileUploader > div:first-child > div:first-child:hover {
            border: 3px dashed #cc0000;
            background-color: #fff8e1;
        }
        .stFileUploader button {
            background: linear-gradient(45deg, #cc0000, #a00000); 
            color: #fff !important;
            border: none;
            border-radius: 15px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: bold;
            box-shadow: 0 3px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .stFileUploader .css-1by9afl p {
            font-size: 1.1rem;
            font-weight: 600;
            color: #d84315 !important;
            text-shadow: none;
        }

        /* CUSTOM CSS PESAN PERINGATAN REKOMENDASI */
        .recommendation-alert {
            padding: 1.5rem;
            border: 3px solid #ffcc00; 
            border-radius: 15px;
            background: linear-gradient(45deg, #fff3e0, #ffecb3); 
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-out;
        }

        .recommendation-alert p {
            margin: 0;
            color: #e65100 !important; 
            font-size: 1.25rem;
            font-weight: 600;
            text-shadow: none;
        }
        
        .recommendation-alert .icon {
            font-size: 1.8rem;
            margin-right: 10px;
            color: #cc0000 !important; 
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("<h1 class='main-title'>Pijjahut</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selamat datang di restoran pizza terbaik. Deteksi piring dan gelas Anda, klasifikasikan gambar pizza, dan dapatkan rekomendasi menu spesial.</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== INITIALIZE SESSION STATE ==========================
# Inisialisasi semua state yang diperlukan
if 'classification' not in st.session_state:
    st.session_state['classification'] = 'none'

if 'detection_result_img' not in st.session_state:
    st.session_state['detection_result_img'] = None
if 'classification_final_result' not in st.session_state:
    st.session_state['classification_final_result'] = None
if 'classification_image_input' not in st.session_state:
    st.session_state['classification_image_input'] = None

# Tambahkan state tracker untuk uploader (untuk kontrol flicker yang lebih baik)
if 'last_yolo_uploader' not in st.session_state:
    st.session_state['last_yolo_uploader'] = None
if 'last_classify_uploader' not in st.session_state:
    st.session_state['last_classify_uploader'] = None

# ========================== UTILITY FUNCTIONS (Load Models) ==========================
# Menggunakan st.cache_resource untuk objek berat seperti model ML
@st.cache_resource
def load_yolo_model(path):
    try:
        # PENTING: Ganti path model Anda jika berbeda
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO dari '{path}'. Pastikan file model ada di lokasi tersebut. Error: {e}")
        return None

@st.cache_resource
def load_classification_model():
    try:
        model = tf.keras.applications.ResNet50(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model Klasifikasi: {e}. Pastikan Anda memiliki koneksi internet dan instalasi TensorFlow/Keras benar.")
        return None

# ========================== KONTROL STATE SAAT BERPINDAH TAB ==========================
# Fungsi untuk membersihkan hasil dari tab yang TIDAK aktif (meminimalkan flicker)
# Fungsi ini dipanggil di awal setiap tab.
def clear_inactive_results(current_tab_index):
    # Tab Deteksi Objek (Index 1)
    if current_tab_index != 1 and st.session_state.get('detection_result_img') is not None:
        st.session_state['detection_result_img'] = None

    # Tab Klasifikasi Gambar (Index 2)
    if current_tab_index != 2:
        if st.session_state.get('classification_final_result') is not None:
            st.session_state['classification_final_result'] = None
        if st.session_state.get('classification_image_input') is not None:
            st.session_state['classification_image_input'] = None


# ========================== HORIZONTAL NAVIGATION (Tabs at Top) ==========================
tabs = st.tabs(["Beranda 🏠", "Deteksi Objek 🍽", "Klasifikasi Gambar 🍕", "Menu Rekomendasi 🌟", "Kontak Kami 📞", "Tentang Kami ℹ"])

# ========================== MAIN CONTENT BASED ON TABS ==========================

# ----------------- BERANDA -----------------
with tabs[0]:
    clear_inactive_results(0)
    st.markdown("<h2 class='section-title'>Jelajahi Inovasi AI Kuliner</h2>", unsafe_allow_html=True)
    
    # KARTU SAMBUTAN
    st.markdown(f"""
    <div class='card' style='background: linear-gradient(180deg, #ffeedd 0%, #fffaf0 100%); border-color: #ff9800;'>
        <p style='font-size: 1.5rem; text-align: center; color: #3e2723;'>
            Kami menyajikan pengalaman kuliner masa depan dengan integrasi <span style='font-weight: bold; color: #cc0000;'>Kecerdasan Buatan (AI)</span>.
            Temukan sajian terbaik, mulai dari deteksi peralatan makan hingga rekomendasi menu personal.
        </p>
        <p style='font-size: 1.1rem; text-align: center; font-style: italic; color: #5d4037;'>
            Satu gigitan, satu algoritma yang sempurna.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("<h3 style='text-align: center; color: #5d4037; font-family: Pacifico, cursive; font-size: 2.2rem; margin-bottom: 1.5rem;'>Fitur Canggih Kami</h3>", unsafe_allow_html=True)
    
    # Bagian Fitur Canggih (3 Kolom)
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 2rem;'>🍽</p>
            <span style='font-weight: bold; color: #ff5722;'>Deteksi Objek Meja</span>
            <p style='font-size: 0.95rem; margin-top: 0.5rem;'>Mengenali piring dan gelas di meja Anda secara instan menggunakan model <span style='font-weight: bold;'>YOLO</span>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat2:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 2rem;'>✨</p>
            <span style='font-weight: bold; color: #ff5722;'>Klasifikasi Pizza Akurat</span>
            <p style='font-size: 0.95rem; margin-top: 0.5rem;'>Menganalisis gambar untuk mengidentifikasi apakah itu pizza atau bukan dengan <span style='font-weight: bold;'>ResNet50</span>.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_feat3:
        st.markdown("""
        <div class='menu-item'>
            <p style='font-size: 2rem;'>🎁</p>
            <span style='font-weight: bold; color: #ff5722;'>Rekomendasi Personal</span>
            <p style='font-size: 0.95rem; margin-top: 0.5rem;'>Dapatkan saran menu spesial yang disesuaikan berdasarkan hasil klasifikasi Anda.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True) 
    

# ----------------- DETEKSI OBJEK -----------------
with tabs[1]:
    clear_inactive_results(1)
    st.markdown("<h2 class='section-title'>Deteksi Objek di Meja Makan 🍽</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Kami menggunakan model <span style='font-weight: bold;'>YOLO (You Only Look Once)</span> yang canggih untuk mengidentifikasi <span style='font-weight: bold;'>Piring</span> dan <span style='font-weight: bold;'>Gelas</span>. Coba upload gambar peralatan makan Anda, dan saksikan AI kami bekerja!</p>
    </div>
    """, unsafe_allow_html=True)
    
    YOLO_MODEL_PATH = 'model/Balgis Isaura_Laporan 4.pt'
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    
    if yolo_model:
        
        col_input_deteksi, col_output_deteksi = st.columns(2) 

        # --- Bagian Input ---
        with col_input_deteksi:
            # Menggunakan key "yolo_uploader"
            uploaded_file_deteksi = st.file_uploader("Upload Gambar Piring atau Gelas (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="yolo_uploader")
            
            # PENTING: Logika reset hasil jika file baru diupload atau dihapus
            if st.session_state.get('last_yolo_uploader') != uploaded_file_deteksi:
                st.session_state['detection_result_img'] = None
                st.session_state['last_yolo_uploader'] = uploaded_file_deteksi # Update tracker

            if uploaded_file_deteksi:
                image = Image.open(uploaded_file_deteksi)
                st.image(image, caption="Gambar Input Anda", use_container_width=True)

                if st.button("Deteksi Sekarang 🚀", type="primary", key="detect_obj"):
                    with st.spinner("⏳ Memproses deteksi objek dengan YOLO..."):
                        try:
                            # Model inference
                            results = yolo_model(image)
                            result_img = results[0].plot()  
                            # Ubah dari BGR (YOLO default) ke RGB (PIL/Streamlit default)
                            result_img_rgb = Image.fromarray(result_img[..., ::-1])
                            
                            st.session_state['detection_result_img'] = result_img_rgb
                            st.success("Deteksi berhasil! Objek piring/gelas telah ditandai.")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat deteksi: {e}. Pastikan format gambar dan model benar.")

        # --- Bagian Output ---
        with col_output_deteksi:
            if st.session_state.get('detection_result_img') is not None:
                st.image(st.session_state['detection_result_img'], caption="Hasil Deteksi YOLO", use_container_width=True)
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #ff5722; border-radius: 15px; text-align: center; padding-top: 100px; color: #ff5722; font-weight: bold;'>HASIL DETEKSI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Model YOLO tidak dapat dimuat dari '{YOLO_MODEL_PATH}'. Pastikan file tersedia.")


# ----------------- KLASIFIKASI GAMBAR -----------------
with tabs[2]:
    clear_inactive_results(2)
    st.markdown("<h2 class='section-title'>Klasifikasi Gambar Pizza 🍕</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p>Bingung apakah yang Anda lihat adalah pizza? Upload gambarnya! Model klasifikasi berbasis <span style='font-weight: bold;'>ResNet50</span> kami akan memberi tahu Anda. Hasil ini akan menentukan rekomendasi menu spesial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    classification_model = load_classification_model()
    
    if classification_model:
        
        col_class_input, col_class_output = st.columns(2)

        # --- Bagian Input ---
        with col_class_input:
            uploaded_file_class = st.file_uploader("Upload Gambar untuk Klasifikasi (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"], key="classify_uploader")

            # PENTING: Logika reset hasil jika file baru diupload atau dihapus
            if st.session_state.get('last_classify_uploader') != uploaded_file_class:
                st.session_state['classification_final_result'] = None
                st.session_state['classification_image_input'] = None
                st.session_state['last_classify_uploader'] = uploaded_file_class # Update tracker

            if uploaded_file_class:
                image_pil = Image.open(uploaded_file_class)
                image_class_resized = image_pil.resize((224, 224))
                
                st.session_state['classification_image_input'] = image_class_resized
                
                st.image(st.session_state['classification_image_input'], caption="Gambar Input Anda (diresize ke 224x224)", use_container_width=True)

                if st.button("Klasifikasikan Sekarang 🔍", type="primary", key="classify_btn"):
                    with st.spinner("⏳ Mengklasifikasikan gambar dengan ResNet50..."):
                        try:
                            img_array = np.array(image_class_resized)
                            if img_array.ndim == 2:
                                img_array = np.stack((img_array,)*3, axis=-1)
                            if img_array.shape[2] == 4:
                                img_array = img_array[:,:,:3] 

                            preprocessed_img = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))
                            
                            predictions = classification_model.predict(preprocessed_img)
                            decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0] 
                            
                            is_pizza = False
                            # Gunakan keywords yang lebih luas dari ImageNet untuk makanan utama
                            pizza_keywords = ['pizza', 'cheese_pizza', 'hot_dog', 'bagel', 'focaccia', 'quiche'] 
                            
                            for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
                                if any(keyword in label.lower() for keyword in pizza_keywords):
                                    is_pizza = True
                                    
                            
                            if is_pizza:
                                final_result = "Pizza"
                                st.session_state['classification'] = 'pizza'
                                st.balloons()
                            else:
                                final_result = "Bukan Pizza"
                                st.session_state['classification'] = 'not_pizza'
                                st.snow()
                            
                            st.session_state['classification_final_result'] = final_result
                            
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

        # --- Bagian Output ---
        with col_class_output:
            if st.session_state.get('classification_final_result') is not None:
                final_result = st.session_state['classification_final_result']
                
                st.markdown("### Hasil Klasifikasi AI")
                
                if final_result == "Pizza":
                    st.success(f"🎉 Selamat! Objek ini terklasifikasi sebagai {final_result}.")
                else:
                    st.info(f"😕 Objek ini terklasifikasi sebagai {final_result} (mungkin makanan atau masakan lain).")

                st.markdown(f"---")
                st.markdown(f"<p style='font-size: 1.8rem; text-align: center; font-weight: bold; color: #cc0000;'>Kesimpulan AI: {final_result}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 300px; border: 2px dashed #ff5722; border-radius: 15px; text-align: center; padding-top: 100px; color: #ff5722; font-weight: bold;'>HASIL KLASIFIKASI AKAN MUNCUL DI SINI</div>", unsafe_allow_html=True)

    else:
        st.warning("Model Klasifikasi (ResNet50) tidak dapat dimuat. Pastikan TensorFlow terinstal dengan benar.")


# ----------------- MENU REKOMENDASI -----------------
with tabs[3]:
    clear_inactive_results(3)
    st.markdown("<h2 class='section-title'>Rekomendasi Menu Spesial 🌟</h2>", unsafe_allow_html=True)
    # KOREKSI: Pastikan sintaks HTML di sini benar
    st.markdown("<p style='text-align: center;'>Rekomendasi ini didasarkan pada hasil klasifikasi gambar Anda di tab sebelumnya. Mari kita lihat apa yang cocok untuk Anda!</p>", unsafe_allow_html=True)
    
    menu = {
        'pizza_spesial': [
            {'nama': 'Pizza Margherita Klasik', 'deskripsi': 'Saus tomat otentik, Mozzarella segar, dan daun Basil. Kesempurnaan Italia.', 'harga': 'Rp 85.000'},
            {'nama': 'Pizza Pepperoni AI', 'deskripsi': 'Daging pepperoni premium dan keju yang dideteksi AI, dijamin terbaik.', 'harga': 'Rp 95.000'}
        ],
        'non_pizza_spesial': [
            {'nama': 'Spaghetti Bolognese Pijjahut', 'deskripsi': 'Spaghetti al dente dengan saus daging rahasia yang kaya rasa.', 'harga': 'Rp 65.000'},
            {'nama': 'Caesar Salad Segar', 'deskripsi': 'Salad sehat dengan ayam panggang, crouton, dan dressing Caesar creamy.', 'harga': 'Rp 55.000'}
        ],
        'dessert_minuman': [
            {'nama': 'Lava Cake Cokelat Panas', 'deskripsi': 'Dessert wajib untuk menutup hidangan Anda.', 'harga': 'Rp 40.000'},
            {'nama': 'Es Teh Lemon Segar', 'deskripsi': 'Pendingin yang sempurna setelah makan pizza pedas.', 'harga': 'Rp 20.000'}
        ]
    }
    
    col_rec1, col_rec2 = st.columns(2)
    
    if st.session_state['classification'] == 'pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #ffe0b2, #ffcc80); border-color: #ff9800;'>
            <p style='font-size: 1.5rem; text-align: center; color: #d84315; font-weight: bold;'>🎉 Gambar Anda adalah <span style='font-weight: bold;'>PIZZA</span>! 🎉</p>
            <p style='font-size: 1.1rem; text-align: center;'>Karena Anda suka pizza, kami rekomendasikan untuk mencoba varian lain atau pendamping yang pas!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🍕 Varian Pizza Wajib Coba")
            for item in menu['pizza_spesial']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("### 🍹 Minuman & Dessert")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
    elif st.session_state['classification'] == 'not_pizza':
        st.markdown("""
        <div class='card' style='background: linear-gradient(45deg, #e1f5fe, #b3e5fc); border-color: #0288d1;'>
            <p style='font-size: 1.5rem; text-align: center; color: #01579b; font-weight: bold;'>🥗 Gambar Anda <span style='font-weight: bold;'>BUKAN PIZZA</span>!</p>
            <p style='font-size: 1.1rem; text-align: center;'>Jika Anda mencari alternatif selain pizza, coba menu non-pizza andalan kami. Dijamin tidak kalah lezat!</p>
        </div>
        """, unsafe_allow_html=True)
        
        with col_rec1:
            st.markdown("### 🍝 Pilihan Non-Pizza Terbaik")
            for item in menu['non_pizza_spesial']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("### 🍰 Dessert untuk Melengkapi")
            for item in menu['dessert_minuman']:
                st.markdown(f"<div class='menu-item'><span style='font-weight: bold;'>{item['nama']}</span> <br> <span style='font-size: 0.9rem;'>{item['deskripsi']}</span> <br> <span style='color:#ff5722; font-weight: bold;'>{item['harga']}</span></div>", unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class='recommendation-alert'>
            <p>
                <span class='icon'>🍕</span> 
                <span style='font-weight: bold;'>Silakan lakukan Klasifikasi Gambar (Tab ke-3) terlebih dahulu</span> untuk mendapatkan rekomendasi menu personal yang paling akurat.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ----------------- KONTAK KAMI -----------------
with tabs[4]:
    clear_inactive_results(4)
    st.markdown("<h2 class='section-title'>Hubungi Kami 📞</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='font-size: 1.2rem; text-align: center;'>Ada pertanyaan, masukan, atau ingin memesan langsung? Jangan ragu untuk menghubungi tim Pijjahut.</p>
        <div class='contact-info'>
            <p><span style='font-weight: bold;'>📍 Alamat:</span> Jl. Digitalisasi No. 101, Kota Streamlit, Kode Pos 404 </p>
            <p><span style='font-weight: bold;'>📞 Telepon:</span> (021) 123-PIZZA (74992)</p>
            <p><span style='font-weight: bold;'>📧 Email:</span> <a href='mailto:pijjahut.ai@gmail.com' style='color: #cc0000 !important; text-decoration: none;'>pijjahut.ai@gmail.com</a></p>
            <p><span style='font-weight: bold;'>🕒 Jam Buka:</span> Setiap Hari, 10:00 - 22:00 WIB</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

# ----------------- TENTANG KAMI -----------------
with tabs[5]:
    clear_inactive_results(5)
    st.markdown("<h2 class='section-title'>Tentang Pijjahut ℹ</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card'>
        <p style='font-size: 1.4rem;'>Pijjahut didirikan dengan visi untuk membawa teknologi <span style='font-weight: bold;'>Kecerdasan Buatan</span> ke ranah kuliner. Kami percaya bahwa data dan algoritma dapat meningkatkan pengalaman bersantap Anda.</p>
        <p>Proyek ini dikembangkan oleh <span style='font-weight: bold;'>Dini Arifatul Nasywa</span> sebagai bagian dari eksplorasi pada topik:</p>
        <ul>
            <li><span style='font-weight: bold;'>Deteksi Objek (YOLOv8):</span> Digunakan untuk mengenali peralatan makan dasar, piring dan gelas.</li>
            <li><span style='font-weight: bold;'>Klasifikasi Gambar (ResNet50):</span> Dimanfaatkan untuk mengidentifikasi produk utama kami: Pizza.</li>
            <li><span style='font-weight: bold;'>Platform:</span> Dibangun menggunakan <span style='font-weight: bold;'>Streamlit</span> untuk tampilan antarmuka yang interaktif dan <span style='font-weight: bold;'>user-friendly</span>.</li>
        </ul>
        <p>Terima kasih telah menjadi bagian dari perjalanan inovatif ini!</p>
        <div style='text-align: center; margin-top: 2rem;'>
            <p style='font-style: italic; color: #cc0000;'>#AIxKuliner #Pijjahut #DiniArifatulNasywa</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ========================== FOOTER ==========================
st.markdown("---")
st.markdown("<p class='footer'>© 2024 Pijjahut. Dibuat dengan 🍕 dan ❤ oleh Dini Arifatul Nasywa.</p>", unsafe_allow_html=True)
