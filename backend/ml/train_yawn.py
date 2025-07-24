import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ---------------------------
# DATASET INSTRUCTIONS
# ---------------------------
# Download the Yawn Eye Dataset from:
# https://www.kaggle.com/datasets/monu999/yawn-eye-dataset-new
# Unzip and place the 'yawn' and 'no_yawn' folders in:
#   backend/ml/datasets/yawn/yawn/
#   backend/ml/datasets/yawn/no_yawn/
# Each folder should contain cropped mouth images (jpg/png).

class YawnDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        for label, subfolder in enumerate(['no_yawn', 'yawn']):
            folder = os.path.join(root_dir, subfolder)
            for dirpath, _, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(dirpath, fname))
                        self.labels.append(label)  # 0=no_yawn, 1=yawn
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0
        return img, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

class YawnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 2)  # no_yawn/yawn
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    print("Started Training Yawn Detection Model...")
    data_dir = 'backend/ml/datasets/yawn'  # Updated path for project root
    batch_size = 32
    epochs = 10
    lr = 0.001
    val_split = 0.2
    transform = get_transforms()
    dataset = YawnDataset(data_dir, transform=transform)
    print(f"Found {len(dataset)} images in dataset.")
    if len(dataset) == 0:
        print("ERROR: No images found! Check your dataset path and folder structure.")
        return
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = YawnCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx+1) % 5 == 0 or (batch_idx+1) == len(train_loader):
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(train_loader)
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}")
    torch.save(model.state_dict(), 'yawn_cnn.pth')
    print("Training complete. Model saved as yawn_cnn.pth")

if __name__ == '__main__':
    train()

"""
README INSTRUCTIONS:
- Download the Yawn Eye Dataset and place 'yawn' and 'no_yawn' folders in backend/ml/datasets/yawn/
- Run: python train_yawn.py
- The trained model will be saved as yawn_cnn.pth for backend inference.
""" 