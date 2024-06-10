import tensorflow as tf
from  tensorflow import keras
from  tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

model=load_model('ImageClassify.keras')

width=180
height=180

st.header("Image Classification Model ")
data_cat=['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


# image=st.text_input('Enter image name','watermelon.jpg')
upload_image=st.file_uploader("Choose An Image ",type=['jpg','jpeg','png'])
if upload_image is not None:
    # image=Image.open(upload_image)
    # st.image(image,caption="Uploaded Image",use_column_width=True)
    st.write("Image Uploaded Successfully")
    image_load=tf.keras.utils.load_img(upload_image,target_size=(height,width))
    img_arr=tf.keras.utils.array_to_img(image_load)
    img_bat=tf.expand_dims(img_arr,0)


    predict=model.predict(img_bat)
    score=tf.nn.softmax(predict)
    st.image(upload_image,width=200)
    st.write("Veg/Fruit in Image is {} with accuracy of {:0.2f}".format(data_cat[np.argmax(score)],np.max(score)*100))
else:
    st.write("No Image is Uploaded")


