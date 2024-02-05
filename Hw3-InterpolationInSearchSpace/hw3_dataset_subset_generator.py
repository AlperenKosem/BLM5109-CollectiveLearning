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
import random

# Cihaz kullanılabilirse GPU'yu kullan, yoksa CPU'yu kullan
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CIFAR-10 veri setini yükleniyor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

print(type(train_set))
print(train_set)

total_data_count = 2000
random_indices = random.sample(range(len(train_set)), total_data_count)
subset = torch.utils.data.Subset(train_set, random_indices)

# Veri kümesini beşe bölelim
split_sizes = [500, 500, 500, 100, 400]  # Her parçanın istenen örnek sayısı
subset_splits = random_split(subset, split_sizes)


print(subset_splits)
for idx, subset_split in enumerate(subset_splits):
    print(f"Parça {idx + 1} Boyutu: {len(subset_split)} örnek")
    


fig, axs = plt.subplots(1, len(subset_splits), figsize=(15, 5))

for i, subset_split in enumerate(subset_splits):
    labels = [data[1] for data in subset_split]  

    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    ax = axs[i] if len(subset_splits) > 1 else axs  # Tek bir subplot varsa
    ax.bar(label_counts.keys(), label_counts.values(), tick_label=train_set.classes)
    ax.set_xlabel('Etiketler')
    ax.set_ylabel('Veri Sayısı')
    ax.set_title(f'Parça {i+1} Etiket')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'Subsets.png')

plt.show()

file_name = "subset_splits.pkl"

with open(file_name, 'wb') as file:
    pickle.dump(subset_splits, file)
