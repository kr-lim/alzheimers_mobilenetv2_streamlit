import streamlit as st
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
from PIL import Image, ImageOps
import cv2

st.header("Alzheimer's Disease Prediction")
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the patient's mri image.")
st.write("This application uses MobileNetV2")


def load_model():
    # load model
    model = keras.models.Sequential()
    mobilenet = MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # freezing layers
    for layer in mobilenet.layers:
        layer.trainable = False

    model.add(mobilenet)
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(4, activation='softmax'))
    model.summary()
    #model.load_weights(r"MobileNetV2_weights.h5")

    return model


model = load_model()

file = st.file_uploader("Please upload an mri image.", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("No image file has been uploaded.")
else:
    image = Image.open(file)
    predictions = import_and_predict(image, model)
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    string = "The patient is predicted to be: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.image(image)
