from typing import List

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf
import keras

import streamlit as st

def disp_img(image_num, dataset):
  fig, ax = plt.subplots()

  if dataset == "training":
    img = X_train[image_num]
  elif dataset == "validation":
    img = X_val[image_num]
  elif dataset == "testing":
    img = X_test[image_num]

  ax.imshow(img.squeeze())

  disease_num = y_train[image_num]

  if disease_num == 0:
    disease_name = "Doesn't have diabetic retinopathy."
  elif disease_num == 1:
    disease_name = "Has stage 1 diabetic retinopathy."
  elif disease_num == 2:
    disease_name = "Has stage 2 diabetic retinopathy."
  elif disease_num == 3:
    disease_name = "Has stage 3 diabetic retinopathy."
  elif disease_num == 4:
    disease_name = "Has stage 4 diabetic retinopathy."

  plt.title(f"Image #{image_num+1} of the {dataset} dataset.\n {disease_name}")

  ax.axis("off")

  st.pyplot(fig)

model = keras.models.load_model('CNN_Model.keras')

data = np.load("retinamnist_128.npz")

#Data split
X_train = data["train_images"]
y_train = data["train_labels"]

X_val = data["val_images"]
y_val = data["val_labels"]

X_test = data["test_images"]
y_test = data["test_labels"]

X_train = X_train[..., [0,1,2]] / 255.0
X_val = X_val[..., [0,1,2]] / 255.0
X_test = X_test[..., [0,1,2]] / 255.0

st.write("""
# Detecting Diabetic Retinopathy with Machine Learning - Live Demonstration
#### By **CaribPixelators**
""")

st.write("""
Pick a portion of the dataset and the image number! 
""")

dataset_choice = st.radio('Pick a portion of the dataset:', ["Training", "Validation", "Testing"])

if dataset_choice == "Training":
  img_number = st.slider('Select an image to test: ', min_value=1, max_value=len(X_train))
  img = disp_img(img_number-1, "training")
  img_probs = model.predict(X_train[img_number-1:img_number+1])[0]
elif dataset_choice == "Validation":
  img_number = st.slider('Select an image to test: ', min_value=1, max_value=len(X_val))
  img = disp_img(img_number-1, "validation")
  img_probs = model.predict(X_val[img_number-1:img_number+1])[0]
elif dataset_choice == "Testing":
  img_number = st.slider('Select an image to test: ', min_value=1, max_value=len(X_val))
  img = disp_img(img_number-1, "testing")
  img_probs = model.predict(X_test[img_number-1:img_number+1])[0]

st.write("""
## Model Predictions
""")

st.progress(text=f"Chance of not having the disease: {str(round(img_probs[0]*100, 2))}%", value=float(img_probs[0]))
st.progress(text=f"Chance of having stage 1 of the disease: {str(round(img_probs[1]*100, 2))}%", value=float(img_probs[1]))
st.progress(text=f"Chance of having stage 2 of the disease: {str(round(img_probs[2]*100, 2))}%", value=float(img_probs[2]))
st.progress(text=f"Chance of having stage 3 of the disease: {str(round(img_probs[3]*100, 2))}%", value=float(img_probs[3]))
st.progress(text=f"Chance of having stage 4 of the disease: {str(round(img_probs[4]*100, 2))}%", value=float(img_probs[4]))

st.write("We hope you enjoyed the demo! Want to see our code? Head to our github page: https://github.com/magocoder-yo/detecting-diabetic-retinopathy-caribpixelators")