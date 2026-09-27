import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

print("Imports completed successfully.")

#Device config
if torch.cuda.is_available():
  device = torch.device('cuda')
  print("Using GPU")
else:
  device = torch.device('cpu')
  print("Using CPU")

class MainDataset(Dataset):
  def __init__(self, split):
    data = np.load("retinamnist_128.npz")

    self.images = data[f"{split}_images"]
    self.labels = data[f"{split}_labels"]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    label = self.labels[idx]

    #Convert format from HWC -> CHW
    image = torch.from_numpy(image).permute(2, 0, 1)

    image = image.float() / 255.0

    label = torch.tensor(label, dtype=torch.long).squeeze()

    return image, label



train_dataset = MainDataset("train")
val_dataset = MainDataset("val")
test_dataset = MainDataset("test")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Data preprocessing completed successfully.")

model = nn.Sequential(
  nn.Conv2d(3, 32, 3, padding=1),
  nn.ReLU(),
  nn.MaxPool2d(2), #shrinks to 64 x 64

  nn.Conv2d(32, 64, 3, padding=1),
  nn.ReLU(),
  nn.MaxPool2d(2), #shrinks to 32 x 32

  nn.Conv2d(64, 128, 2, padding=1),
  nn.ReLU(),
  nn.MaxPool2d(2), #shrinks to 16 x 16

  nn.Conv2d(128, 256, 2, padding=1),
  nn.ReLU(),
  nn.MaxPool2d(2), #shrinks to 8x8

  nn.Flatten(),
  nn.Linear(256 * 8 * 8, 128), #converts pixel channels to 128 channels total
  nn.ReLU(),

  nn.Linear(128, 5)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

"""
estop= EarlyStopping(monitor = 'val_loss',patience = 10,restore_best_weights=True)
model.fit(X_train,y_train_hotcode,validation_data = (X_val,y_val_hotcode),batch_size = 32,epochs = 100,callbacks = [estop])
"""

for epoch in range(20):
  model.train()

  for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

  print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")

print("Model built successfully.")

torch.save(model.state_dict(), "model_weights.pth")
print("Model saved successfully.")