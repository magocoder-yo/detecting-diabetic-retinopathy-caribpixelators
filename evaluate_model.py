import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

import tensorflow as tf
import keras

model = keras.models.load_model("CNN_Model.keras")

data = np.load("retinamnist_128.npz")

X_test = data['test_images']
y_test = data['test_labels']

X_test = X_test[..., [0,1,2]] / 255.0

pred = np.argmax(model.predict(X_test), axis=1)

evaluating_text = f"""Accuracy Score: {accuracy_score(y_test, pred)}

Classification Report:
{classification_report(y_test, pred)}
"""

with open("./model-diagnostics/model_diagnostic_information.txt", "w", encoding="utf-8") as file:
    file.write(evaluating_text)

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"], cmap="Blues")
plt.title("Confusion Matrix Display")
plt.savefig("./model-diagnostics/confusion-matrix.png")

print("Evaluation complete.")