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
    




# model_weights_path = 'wi_weights_common.pth'
# model_weights_path = 'w1_weights_m1.pth'
# model_weights_path = 'w2_weights_m2.pth'
# model_weights_path = 'wL_weights_searching_line.pth'
# model_weights_path = 'wU_weights_searching_triangle.pth'
# model_weights_path = 'wF_weights_m1_m2.pth'

#model_weights_path = 'wL_weights_searching_line_ext.pth'
model_weights_path = 'wU_weights_searching_triangle_ext.pth'








f = open(f'reports/report_for_{model_weights_path}.txt', "a")

model = SimpleCNN()  # Varsayılan bir model oluşturulur
model.load_state_dict(torch.load(f'weights/{model_weights_path}'))


test_loader = DataLoader(subset_splits[4], batch_size=8, shuffle=False)



# Test veri kümesini kullanarak modelin tahminlerini ve gerçek etiketleri al
model.eval()  # Modeli değerlendirme moduna geçir
true_labels = []
predicted_labels = []

with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırak
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        true_labels.extend(labels.numpy())  # Gerçek etiketler
        predicted_labels.extend(predicted.numpy())  # Modelin tahminleri

# Sınıflandırma raporunu oluştur

report = classification_report(true_labels, predicted_labels)
f.write(report)
print(report)