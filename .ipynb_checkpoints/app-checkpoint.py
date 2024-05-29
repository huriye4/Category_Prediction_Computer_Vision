import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array
from predict import predict, get_categories, preprocess_image, get_random_image, project_introduction, project_phases, libraries_used, conclusion




def main():
    st.markdown(
        """
        <link rel="stylesheet" href="style.css">
        """,
        unsafe_allow_html=True )

    # Model ve diğer sabitler
    MODEL_PATH = "mobilenetv2_18K_model2.h5"
    STATIC_DIR = "static"
    CATEGORIES = get_categories()

    # Modeli yükle
    def my_load_model(model_path):
        return load_model(model_path)

    model = my_load_model(MODEL_PATH)

    # Sidebar
    st.sidebar.title("IME Brand & Techpro Education")
    # Resmi sidebar'a ekle
    # Resmin yolunu belirtin
    image_path = "brand_logo.jpeg"

    # Orijinal resmi açın
    original_image = Image.open(image_path)

    # Resmi gösterin
    st.sidebar.image(original_image, width=200)
    # Sayfa seçim radyo butonları
    page = st.sidebar.radio("Select Page", ["Home", "Prediction"])

    # Home Sayfası
    if page == "Home":
        st.title("Welcome to the Category Prediction Project")

        # Sayfaları dinamik olarak göstermek için bir seçim kutusu ekleyin
        page_choice = st.sidebar.radio("Select Page", ["Project Introduction", "Project Phases", "Libraries Used", "Conclusion"])

        # Seçime göre doğru sayfayı gösterin
        if page_choice == "Project Introduction":
            project_introduction()
        elif page_choice == "Project Phases":
            project_phases()
        elif page_choice == "Libraries Used":
            libraries_used()
        elif page_choice == "Conclusion":
            conclusion()

    # Prediction Sayfası
    elif page == "Prediction":
        st.title("Prediction Project")

        st.write("Please choose one of the two options:")
        option = st.radio("", ["Upload an image", "Generate a random image"], index=0)

        if option == "Upload an image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_path = os.path.splitext(uploaded_file.name)[0]  # Dosya adını ve uzantısını ayırın
                category = os.path.basename(os.path.dirname(image_path))  # Dosya yolundan klasör adını alın
                st.image(image, caption=f'Selected Image from Category: {category}', width=300)

                if st.button("Make Prediction"):
                    predictions = predict(image, model, CATEGORIES)
                    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
                    df_predictions = pd.DataFrame(sorted_predictions, columns=["Category", "Confidence"])
                    st.write("Predictions:")
                    styled_df = df_predictions.style.format({"Confidence": "{:.2f}%"})
                    st.dataframe(styled_df)

        # User chooses the option to generate a random image
        elif option == "Generate a random image":
            if st.button("Get Random Image"):
                random_image_path, category = get_random_image(STATIC_DIR)
                image = Image.open(random_image_path)
                st.image(image, caption=f'Selected Image from Category: {category}', width=300)
                st.session_state['random_image'] = image  # Resmi session state'e kaydediyoruz

            if st.button("Prediction"):
                if 'random_image' in st.session_state:
                    image = st.session_state['random_image']  # Session state'ten resmi alıyoruz
                    prediction = predict(image, model, CATEGORIES)
                    sorted_prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
                    df_prediction = pd.DataFrame(sorted_prediction, columns=["Category", "Confidence"])
                    st.write("Prediction:")
                    styled_df = df_prediction.style.format({"Confidence": "{:.2f}%"})
                    st.dataframe(styled_df)

if __name__ == "__main__":
    main()

