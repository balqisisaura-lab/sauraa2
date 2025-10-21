# ==========================
# ResNet50 - Klasifikasi RPS
# ==========================
else:
    model = load_resnet_model()
    st.subheader("✋ Jalankan Klasifikasi RPS")

    if st.button("Jalankan Klasifikasi"):
        # Ambil input shape model (misal (None, 150, 150, 3))
        input_shape = model.input_shape[1:3]

        # Resize sesuai input model
        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # tambahkan batch dimension
        img_array = img_array / 255.0  # normalisasi (umum di training RPS)

        # Prediksi
        preds = model.predict(img_array)
        class_names = ["Rock", "Paper", "Scissors"]

        # Hasil prediksi
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds)

        st.success(f"🧩 Prediksi: **{predicted_class}** ({confidence*100:.2f}%)")
