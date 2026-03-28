import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

class AudioDeepfakeDataset(Dataset):
    def __init__(self, data_dir, duration=5, sample_rate=16000):
        self.data_dir = data_dir
        self.duration = duration
        self.sample_rate = sample_rate
        self.audio_files = []
        self.labels = []
        
        classes = ['real', 'fake']
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            if os.path.exists(cls_dir):
                for file in os.listdir(cls_dir):
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        self.audio_files.append(os.path.join(cls_dir, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio using librosa
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        
        # Pad if audio is shorter than target duration
        target_length = self.duration * self.sample_rate
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
        # Extract Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalize and convert to tensor
        S_dB = (S_dB - np.mean(S_dB)) / np.std(S_dB)
        S_tensor = torch.tensor(S_dB, dtype=torch.float32).unsqueeze(0) # Add channel dim (1, n_mels, t)
        
        return S_tensor, label

class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        
        # Calculate linear input size:
        # Mel bins: 128 -> 64 -> 32 -> 16
        # Time steps for 5s @ 16kHz with default hop_length (512): ~157
        # 157 -> 78 -> 39 -> 19
        # Flattened: 128 * 16 * 19 = 38912
        self.fc1 = nn.Linear(128 * 16 * 19, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        out = self.fc2(x)
        return out

def train_audio_model(data_dir, num_epochs=10, batch_size=16, learning_rate=0.001):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        print("Audio dataset directory not found. Creating structure...")
        os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'real'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'fake'), exist_ok=True)
        print("Add audio files to begin training.")
        return

    train_dataset = AudioDeepfakeDataset(train_dir)
    val_dataset = AudioDeepfakeDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting audio training on {device}...")
    best_acc = 0.0

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

        if len(train_dataset) > 0:
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)
            print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
        torch.save(model.state_dict(), '../models/audio_best_model.pth')
        print("Saved best audio model.")

if __name__ == '__main__':
    data_directory = '../datasets/audio'
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    train_audio_model(data_directory)
