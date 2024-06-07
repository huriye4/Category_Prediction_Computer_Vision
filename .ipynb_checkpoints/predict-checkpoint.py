import os
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_categories():
    return os.listdir('static')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img, np.expand_dims(img_array, axis=0)

def get_random_image(static_dir):
    category = np.random.choice(os.listdir(static_dir))
    image_path = os.path.join(static_dir, category, np.random.choice(os.listdir(os.path.join(static_dir, category))))
    return image_path, category

def predict(image, model, categories):
    if isinstance(image, str):  # If the input is a file path
        img, img_array = preprocess_image(image)
    else:  # If the input is an image object
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    sorted_indices = np.argsort(predictions[0])[::-1]  # Büyükten küçüğe sırala
    results = [(categories[i], predictions[0][i] * 100) for i in sorted_indices]
    return results

def project_phases():
    st.write("""
    # Project Phases
        -Data Collection and Web Scraping
        -Data Preprocessing
        -Model Training
        -Model Evaluation
        -Streamlit  Interface Deployment
    """)
    st.image('images/Final Data Set.png', caption='Data Set')
    st.image('images/Train-Val Category.png', caption='İmages Numbers')

# Proje İntroduction
def project_introduction():
    st.write("""
    ## Category Prediction with Computer Vision

    As part of this project, an artificial intelligence model has been developed to predict product categories from the product photos of our e-commerce company.
    """)
    st.image('images/Proje.jpg', caption='Proje Modeli',width =300)

# Libraries Used
def libraries_used():
    st.write("""
    # Libraries Used in Our Project
    - Python: Programming language
    - TensorFlow-Keras: Deep learning library
    - OpenCV: Computer vision library
    - Streamlit: Web application library
    - PIL (Pillow): Image processing library
    - NumPy: Numerical computing library
    - Pandas: Data analysis library
    - Datetime: Date and time handling module
    - os: Operating system interface module
    """)
    

def conclusion():
    st.write("""
    ## Conclusion
    """)
    st.image('images/PREDİCTONS.png', caption='Predictions')
    st.image('images/confusion matrix.jpeg', caption='Proje Modeli')
   
