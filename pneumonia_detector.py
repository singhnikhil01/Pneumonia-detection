import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import cv2 as cv
from streamlit_login_auth_ui.widgets import __login__

__login__obj = __login__(auth_token="courier_auth_token",
                         company_name="Shims",
                         width=200,
                         height=250,
                         logout_button_name='Logout',
                         hide_menu_bool=False,
                         hide_footer_bool=False,
                         lottie_url='https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:
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
            model_url = "https://github.com/Singhsansar/Pneumonia-detection/blob/main/transfer_learning.h5"
            model_path = "transfer_learning.h5"
            interpreter = load_model_from_url(model_url, model_path)
            # interpreter.summary(print_fn=lambda x: st.text(x))
            image = img.convert('RGB')
            image = np.array(image)
            image = cv.resize(image,(224, 224))
            # img_rgb = resized_image.convert('RGB')  # Convert image to RGB format
            image = np.array([image])
            img_array = image / 255.0
            img_array -= img_array.mean()
            img_array /= img_array.std()
            # img_array = np.expand_dims(img_array, axis=0)
              # Normalize the image
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
        model_url = "https://github.com/Singhsansar/Pneumonia-detection/blob/main/transfer_learning.h5"
        model_path = "transfer_learning.h5"
        interpreter = load_model_from_url(model_url, model_path)
        interpreter.summary(print_fn=lambda x: st.text(x))
        
        st.header("The loss plot for validation and train set")
        image1 = Image.open("images/val_train_loss.jpeg")
        st.image(image1)
        st.caption("The orange line represents the validation curve")
        st.caption("The blue line represents the train curve")

        st.header("The accuracy plot for train and test set")
        image2 = Image.open("images/acc_train_test.jpeg")
        st.image(image2)
        
        st.write("Check out this [link](https://github.com/Singhsansar/Pneumonia-detection) to know more about the code.")
        st.write("The [link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) to the dataset we used to train our model.")