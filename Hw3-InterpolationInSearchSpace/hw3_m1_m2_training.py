import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Cihaz kullanılabilirse GPU'yu kullan, yoksa CPU'yu kullan
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_name = "subset_splits.pkl"
with open(file_name, 'rb') as file:
    subset_splits = pickle.load(file)

print(subset_splits)
for idx, subset_split in enumerate(subset_splits):
    print(f"Parça {idx + 1} Boyutu: {len(subset_split)} örnek")
    



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.convLayer1 = nn.Conv2d(3, 32, 5)             # First conv layer (3(input), 32(output), 5(filter_size))
        self.maxPool = nn.MaxPool2d(2, 2)                 # Max Pool (2(filter_size), 2(stride))
        self.convLayer2 = nn.Conv2d(32, 64, 5)            # Second conv layer (32(input), 64(output), 5(filter_size))
        self.drop1 = nn.Dropout(0.2, inplace=False)       # Dropout layer with probability 0.2 
        self.fullyc1 = nn.Linear(1600, 200)               # Fully Connected Layer (64*5*5(input), 200(output))
        self.xav = nn.init.xavier_normal_(self.fullyc1.weight)
        self.fullyc2 = nn.Linear(200, 100)                # Fully Connected Layer (200(input), 100(output))
        self.fullyc3 = nn.Linear(100, 10)                 # Fully Connected Layer (100(input), 10(output))
        

    def forward(self, img):
        img = self.maxPool(self.drop1(F.relu(self.convLayer1(img))))
        img = self.maxPool(self.drop1(F.relu(self.convLayer2(img))))
        # img = self.maxPool(F.relu(self.convLayer1(img)))
        # img = self.maxPool(F.relu(self.convLayer2(img)))
        img = img.view(-1, 64 * 5 * 5)
        img = F.relu(self.fullyc1(img))
        img = F.relu(self.fullyc2(img))
        img = self.fullyc3(img)
        return img   

    

lr = 0.001
num_epochs = 14
batch_size = 8

new_subset = torch.utils.data.ConcatDataset([subset_splits[1], subset_splits[2]])

train_loader = DataLoader(new_subset, batch_size=batch_size, shuffle=True)

print(type(train_loader))

print("Epoch Sayisi: ", num_epochs, "  learning_rate: ", lr)
print("------------------------------------------------------------")


model = SimpleCNN()

model_weights_path = 'weights/wi_weights_common.pth'
model = SimpleCNN()  # Varsayılan bir model oluşturulur
model.load_state_dict(torch.load(model_weights_path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)  # Örnek bir learning rate (öğrenme oranı)

loss_values = []
accuracy_values = []


for epoch in range(num_epochs):
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    # Mini-batch'ler üzerinde eğitim
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    accuracy = correct_predictions / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}')
    loss_values.append(running_loss / len(train_loader))
    accuracy_values.append(accuracy)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_values, label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Eğitim Loss Değerleri')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_values, label='Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Eğitim Accuracy Değerleri')
plt.legend()

model_weights_path = 'wF_weights_m1_m2.pth'
# Modelin ağırlıklarını kaydet
torch.save(model.state_dict(), f'weights/{model_weights_path}')
plt.savefig(f'reports/{model_weights_path}_training.png')
plt.tight_layout()
plt.show()




print('Finished')
