import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Device config
if torch.cuda.is_available():
  device = torch.device('cuda')
  print("Using GPU")
else:
  device = torch.device('cpu')
  print("Using CPU")

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

model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model.eval()

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

test_dataset = MainDataset("test")
test_loader = DataLoader(test_dataset, batch_size=32)

model = model.to(device)

num_correct = 0
num_total = 0

results = []
with torch.no_grad():
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    predictions = outputs.argmax(dim=1)

    num_correct += (predictions == labels).sum().item()
    num_total += labels.size(0)

accuracy = num_correct / num_total

print(f"Accuracy: {accuracy}")