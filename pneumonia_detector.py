import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
import os
import cv2 as cv

st.set_page_config(page_title='PNEUMONIA DETECTION SYSTEM')
st.markdown("<h1 style='text-align: center; color: white;'>PNEUMONIA DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.text("")
tab1,tab12 = st.tabs(["Check for Pneumonia","Know the Model"])
with tab1:
    def main():
        file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
        class_btn = st.button("Classify")

        if file_uploaded is not None:
            img = Image.open(file_uploaded)
            st.image(img, caption='Uploaded Image', use_column_width=True)

        if class_btn:
            if file_uploaded is None:
                st.write("Invalid command, please upload an image")
            else:
                with st.spinner('Model working....'):
                    result = predict(img)
                    st.success('Classified')
                    if result=='Pneumonia Detected':
                        st.error(result)
                    else:
                        st.info(result)

    def predict(img):
        model_path = "transfer_learning.h5"
        interpreter = tf.keras.models.load_model(model_path)
    
        image = img.convert('RGB')
        image = np.array(image)
        image = cv.resize(image,(224, 224))
        image = np.array([image])
        img_array = image / 255.0
        img_array -= img_array.mean()
        img_array /= img_array.std()
        result = interpreter.predict(img_array)

        if result[0][0] > 0.5:
            return 'Pneumonia Detected'
        else:
            return 'No Pneumonia Detected'

    if __name__ == "__main__":
        main()

with tab12:
    st.header("The layers used in the model:")
    st.text(" ")
    model_path = "transfer_learning.h5"
    interpreter = tf.keras.models.load_model(model_path)
    interpreter.summary(print_fn=lambda x: st.text(x))

    st.header("The loss plot for validation and train set")
    image1 = Image.open("image/val_train_loss.jpg")
    st.caption("The orange line represents the validation curve")
    st.caption("The blue line represents the train curve")

    st.header("The accuracy plot for train and test set")
    image2 = Image.open("image/acc_train_test.jpg")
    st.image(image2)
    
    st.write("Check out this [link](https://github.com/Singhsansar/Pneumonia-detection) to know more about the code.")
    st.write("The [link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) to the dataset we used to train our model.")