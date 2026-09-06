import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import torch

print("Imports completed successfully.")

data = np.load("retinamnist_128.npz")

X_train = data["train_images"]
y_train = data["train_labels"]

X_val = data["val_images"]
y_val = data["val_labels"]

X_test = data["test_images"]
y_test = data["test_labels"]

X_train = X_train[..., [0,1,2]] / 255.0
X_val = X_val[..., [0,1,2]] / 255.0
X_test = X_test[..., [0,1,2]] / 255.0

print("Data preprocessing completed successfully.")

def disp_img(image_num, dataset):
  if dataset == "training":
    plt.imshow(X_train[image_num].squeeze())
  elif dataset == "validation":
    plt.imshow(X_val[image_num].squeeze())
  elif dataset == "testing":
    plt.imshow(X_test[image_num].squeeze())

  plt.title(f"Image #{image_num+1} of the {dataset} dataset.\n Disease label: {y_train[image_num]}")

  plt.axis("off")
  plt.show()

disp_img(1, "training")


encoder = LabelEncoder()
y_encode = encoder.fit_transform(y_train)

y_train_hotcode = to_categorical(y_train,5)
y_val_hotcode = to_categorical(y_val,5)
y_test_hotcode = to_categorical(y_test,5)

print("Target data configuration successful.")

model = Sequential([
    Input(shape = (128,128,3)),
    Conv2D(32,3,activation='relu'),
    MaxPooling2D(2), # shrinks to 64x64

    Conv2D(64,3,activation='relu'),
    MaxPooling2D(2), # shrinks to 32x32

    Conv2D(128,2,activation='relu'),
    MaxPooling2D(2), # shrinks to 16x16

    Conv2D(128, 2,activation='relu'),
    MaxPooling2D(2), #shrinks to 8x8

    Flatten(), #converts to a 1D array

    Dense(128,activation='relu'),
    Dropout(0.2),

    Dense(5,activation='softmax')

])
model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate = 5e-5),
    loss="categorical_crossentropy"
)

estop= EarlyStopping(monitor = 'val_loss',patience = 10,restore_best_weights=True)
model.fit(X_train,y_train_hotcode,validation_data = (X_val,y_val_hotcode),batch_size = 32,epochs = 100,callbacks = [estop])

print("Model built successfully.")

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

pred = np.argmax(model.predict(X_test),axis = 1)
print(classification_report(y_test,pred))

ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=encoder.classes_,cmap = 'Blues')
plt.show()

model.save('CNN_Model.keras')
print("Model saved successfully.")