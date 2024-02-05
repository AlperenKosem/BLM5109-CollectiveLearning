import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import psutil
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import time
import pickle
w0 = 0.075
dbfile = open(f'examplePickle_w0:{w0}', 'rb')    
results = pickle.load(dbfile)
dbfile.close()

plt.figure(figsize=(12, 5))

# Loss değerlerini görselleştirme
plt.subplot(1, 2, 1)
for optimizer_name, result in results.items():
    plt.plot( result['losses'], label=optimizer_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.title(f'Loss ()')
plt.legend()

# Accuracy değerlerini görselleştirme
plt.subplot(1, 2, 2)
for optimizer_name, result in results.items():
    plt.plot(result['accuracies'], label=optimizer_name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Accuracy ()')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'Loss_Acc_vs_Epoch_w0 = {w0}.png')


plt.show(block=False)

############## time vs loss vs accuracy
plt.figure(figsize=(12, 5))

# Loss değerlerini görselleştirme
plt.subplot(1, 2, 1)
for optimizer_name, result in results.items():
    plt.plot(result['epoch_times'],result['losses'], label=optimizer_name)
    # plt.xticks(range(len(result['epoch_times'])), result['epoch_times'])
plt.xlabel('time')
plt.ylabel('Loss')
plt.grid(True)
plt.title(f'Loss vs time)')
plt.legend()

# Accuracy değerlerini görselleştirme
plt.subplot(1, 2, 2)
for optimizer_name, result in results.items():
    plt.plot(result['epoch_times'], result['accuracies'], label=optimizer_name)

plt.xlabel('time')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs time ')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'LOSS_ACC_vs_time_w0 = {w0}.png')

plt.figure(figsize=(12, 5))
# Accuracy değerlerini görselleştirme
plt.subplot(1, 2, 1)

for name,result in results.items():
    total_weights_list = result['weights']
    tsne_ = TSNE(n_components=2,perplexity=2).fit_transform(total_weights_list[:26])
    plt.scatter(tsne_[:, 0], tsne_[:, 1], label=name, alpha=0.7)
    for i, txt in enumerate(range(1)):
            plt.annotate(txt, (tsne_[i, 0], tsne_[i, 1]))   


plt.title('t-SNE Visualization')
plt.legend()

plt.subplot(1, 2, 2)

for name,result in results.items():
    total_weights_list = result['weights']

    pca_embedding = PCA(n_components=2, svd_solver='full').fit_transform(total_weights_list[:26])

    plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], label=name, alpha=0.7)
    for i, txt in enumerate(range(1)):
            plt.annotate(txt, (pca_embedding[i, 0], pca_embedding[i, 1]))        

plt.title('PCA Visualization')
plt.legend()
plt.tight_layout()
plt.savefig(f'TSNE_PCA PLOTS_w0 = {w0}.png')

plt.show(block=False)