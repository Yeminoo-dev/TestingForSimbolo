import streamlit as st
import tensorflow as tf
import ViT
import numpy as np
import pandas as pd
from PIL import Image
import urllib

path = 'https://github.com/Yeminoo-dev/Sample-Project/blob/main/ViT_cifar10.h5'
file = urllib.request.urlretrieve(path)

st.set_page_config(page_title = 'Project', layout = 'wide')

st.title("Demo project")
st.write("This is just a demo Vision Transformer trained on Cifar10 dataset without pretraining. Achieving 61% top-1-accuracy and 97% top-5-accuracy after 10 epochs.")

file = st.file_uploader("Choose a file", type = ['png', 'jpg'])

if file is None:
    st.write("Please upload a file")


with st.sidebar:
    options = st.selectbox("Top n predictions",
                           (1, 5))

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_heads = 8
num_layers = 12
patch_size = 6
conv_output = 68
image_size = 72
num_patches = (conv_output // patch_size) ** 2
num_class = 10
projection_dim = 128
x = tf.random.uniform((1, 32, 32, 3))

if file is not None:
    image = Image.open(file)
    st.image(image)

    image = np.array(image)
    image = tf.expand_dims(image, axis = 0)

    model = ViT.ViT_Transformer(num_heads, 
                 num_layers, 
                 patch_size, 
                 num_patches, 
                 num_class, 
                 image_size, 
                 projection_dim, 
                 epsilon = 0.001, 
                 dropout = 0.2)

    model(x)
    model.load_weights(file)
    result = model(image)
    values, indices = tf.math.top_k(result, k = options)
    idx, prob = indices[0].numpy(), values[0].numpy()
    pred = [labels[i] for i in idx]
    df = pd.DataFrame({"Classes" : pred, "Probability" : prob})
    st.write(df)

