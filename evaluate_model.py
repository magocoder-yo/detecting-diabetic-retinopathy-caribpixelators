import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras

model = keras.models.load_model("CNN_Model_128p.keras")

data = np.load("retinamnist_128.npz")

X_test = data['test_images']
y_test = data['test_labels']

X_test = X_test[..., [0,1,2]] / 255.0

pred = np.argmax(model.predict(X_test), axis=1)

print(f"Accuracy Score: {accuracy_score(y_test, pred)}")