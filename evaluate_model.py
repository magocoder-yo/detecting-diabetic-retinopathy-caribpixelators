import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras

model = keras.models.load_model("CNN_Model.keras")

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

pred = np.argmax(model.predict(X_test), axis=1)
y_proba = model.predict(X_test)

evaluating_text = f"""Accuracy Score: {accuracy_score(y_test, pred)}

Classification Report:
{classification_report(y_test, pred)}
"""

with open("./model-diagnostics/model_diagnostic_information.txt", "w", encoding="utf-8") as file:
    file.write(evaluating_text)

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"], cmap="Blues")
plt.title("Confusion Matrix Display")
plt.savefig("./model-diagnostics/confusion-matrix.png")

encoder = LabelEncoder()
y_encode = encoder.fit_transform(y_train)

y_train_hotcode = to_categorical(y_train,5)
y_val_hotcode = to_categorical(y_val,5)
y_test_hotcode = to_categorical(y_test,5)

n_classes = y_test_hotcode.shape[1]

# Compute ROC curve and ROC area for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_hotcode[:, i], y_proba[:, i])
    auc_score = roc_auc_score(y_test_hotcode[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--') # random guessing curve
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Each Class')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("./model-diagnostics/ROC-Curve-Per-Class.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Class Distribution Across Datasets', fontsize=16)

# Training Set
sns.countplot(x=y_train.flatten(), ax=axes[0])
axes[0].set_title('Training Set Class Distribution')
axes[0].set_xlabel('Class Label')
axes[0].set_ylabel('Count')

# Validation Set
sns.countplot(x=y_val.flatten(), ax=axes[1])
axes[1].set_title('Validation Set Class Distribution')
axes[1].set_xlabel('Class Label')
axes[1].set_ylabel('Count')

# Test Set
sns.countplot(x=y_test.flatten(), ax=axes[2])
axes[2].set_title('Test Set Class Distribution')
axes[2].set_xlabel('Class Label')
axes[2].set_ylabel('Count')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./model-diagnostics/Class-Distributions.png")

print("Evaluation complete.")