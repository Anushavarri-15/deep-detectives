import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VideoDeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None, frames_per_video=10):
        self.data_dir = data_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.videos = []
        self.labels = []
        
        # Assume directory structure: train/real, train/fake
        classes = ['real', 'fake']
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            if os.path.exists(cls_dir):
                for file in os.listdir(cls_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        self.videos.append(os.path.join(cls_dir, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames) # Shape: (frames_per_video, C, H, W)
            
        return frames, label

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if frame_count > 0:
            step = max(1, frame_count // self.frames_per_video)
            for i in range(self.frames_per_video):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, frame_count - 1))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL format for torchvision transforms
                    from PIL import Image
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                else:
                    break
        
        # Pad if needed
        from PIL import Image
        while len(frames) < self.frames_per_video:
            frames.append(Image.new('RGB', (224, 224)))
            
        cap.release()
        return frames

class VideoCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoCNN, self).__init__()
        # Use a pretrained ResNet-18 to act as feature extractor for each frame
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove top classification layer
        
        # RNN for temporal sequence
        self.lstm = nn.LSTM(input_size=num_ftrs, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process frames individually
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(c_in)
        
        # Reshape for LSTM
        r_in = features.view(batch_size, seq_len, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        
        # Get output from the last time step
        out = self.fc(r_out[:, -1, :])
        return out

def train_video_model(data_dir, num_epochs=10, batch_size=4, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        print("Video dataset directory not found. Creating structure...")
        os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'fake'), exist_ok=True)
        print("Add videos to begin training.")
        return

    train_dataset = VideoDeepfakeDataset(train_dir, transform=transform)
    val_dataset = VideoDeepfakeDataset(val_dir, transform=transform)

    if len(train_dataset) == 0:
        print("No training videos found.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VideoCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    print(f"Starting video training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Add simpler validation block (omitted loops for brevity)
        torch.save(model.state_dict(), '../models/video_best_model.pth')
        print(f"Saved model to ../models/video_best_model.pth")

if __name__ == '__main__':
    data_directory = '../datasets/videos'
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    train_video_model(data_directory)
