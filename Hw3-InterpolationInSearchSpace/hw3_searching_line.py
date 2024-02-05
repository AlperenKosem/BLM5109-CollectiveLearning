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

    

m1_model_weights_path = 'w1_weights_m1.pth'

model_m1 = SimpleCNN()  # Varsayılan bir model oluşturulur
model_m1.load_state_dict(torch.load(f'weights/{m1_model_weights_path}'))

m2_model_weights_path = 'w2_weights_m2.pth'

model_m2 = SimpleCNN()  # Varsayılan bir model oluşturulur
model_m2.load_state_dict(torch.load(f'weights/{m2_model_weights_path}'))


weights_m1 = model_m1.state_dict()
weights_m2 = model_m2.state_dict()

weights_new = model_m1.state_dict().copy()
# Öncelikle, alfa değerlerini içeren bir aralık belirleyelim
alfas = np.linspace(0, 1, 11)  # 0 ile 1 arasında 11 adet alfa değeri

accuracy_values = []  # Accuracy değerlerini saklamak için bir liste oluşturuyoruz

for alfa in alfas:
    weights_new = model_m1.state_dict().copy()  # Yeni bir ağırlık sözlüğü oluşturuyoruz

    # Ağırlıkları alfa değeri ile birleştirme
    for key in weights_m1.keys():
        if key in weights_m2:
            weights_new[key] = alfa * weights_m1[key] + (1 - alfa) * weights_m2[key]

    # Yeni modeli oluşturup eğitim kümesi üzerinde test edelim
    model_for_searching = SimpleCNN()
    model_for_searching.load_state_dict(weights_new)

    test_loader = DataLoader(subset_splits[3], batch_size=8, shuffle=False)

    model_for_searching.eval()  # Modeli değerlendirme moduna geçir
    true_labels = []
    predicted_labels = []

    with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırak
        for data in test_loader:
            inputs, labels = data
            outputs = model_for_searching(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.numpy())  # Gerçek etiketler
            predicted_labels.extend(predicted.numpy())  # Modelin tahminleri

    # Accuracy değerlerini hesaplayıp listeye ekleyelim
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    accuracy = report['accuracy']
    accuracy_values.append(accuracy)
    print(f'Alfa: {alfa}, Accuracy: {accuracy}')

# Aldığımız accuracy değerlerini görselleştirelim
plt.plot(alfas, accuracy_values, marker='o')
plt.xlabel('Alfa Değeri')
plt.ylabel('Accuracy')
plt.title('Alfa - Accuracy')
plt.grid(True)
plt.savefig(f'reports/w1_w2_searching_accuracies_line.png')

plt.show()

max_accuracy_index = np.argmax(accuracy_values)  
max_accuracy = accuracy_values[max_accuracy_index]  
corresponding_alfa = alfas[max_accuracy_index] 

print(f'En yüksek accuracy değeri: {max_accuracy} (Alfa = {corresponding_alfa})')

# Ağırlıkları alfa değeri ile birleştirme
for key in weights_m1.keys():
    if key in weights_m2:
        weights_new[key] = corresponding_alfa * weights_m1[key] + (1 - corresponding_alfa) * weights_m2[key]

model_for_searching.load_state_dict(weights_new)
model_weights_path = 'wL_weights_searching_line.pth'

# Modelin ağırlıklarını kaydet
torch.save(model_for_searching.state_dict(), f'weights/{model_weights_path}')

