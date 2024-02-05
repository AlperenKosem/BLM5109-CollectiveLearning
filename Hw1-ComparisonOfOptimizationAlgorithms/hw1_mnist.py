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

num_epochs_list = [25] 
learning_rates = [0.01]
batch_size = 512
w0_values = [0.2]

torch.manual_seed(42)


optimizers = {
    'SGD': optim.SGD,
    'SGD_Momentum': optim.SGD,
    'Adam': optim.Adam,
}

results = {}


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=(3, 3), stride=(1, 1), padding = 1, padding_mode='zeros') 
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride=2, padding=0)  # 32 * 14 * 14
        
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=(3, 3), stride=(1, 1), padding = 1, padding_mode='zeros') 
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 5, kernel_size=(3, 3), stride=(1, 1), padding = 1, padding_mode='zeros') 

        self.fc1 = nn.Linear(in_features = 7 * 7 * 5  , out_features = 10, bias=True)
        self.flatten = nn.Flatten()
   
    def initialize_weights(self, w0):
        for layer in [self.conv1, self.conv2, self.fc1]:
            nn.init.normal_(layer.weight, mean=w0, std=0.01)
            nn.init.constant_(layer.bias, 0.01)

    def forward(self, x):

        x = torch.relu(self.conv1(x))    # 28* 28* 1    -> 28 * 28 * 32
        x = self.pool(x)                 # 28 * 28 * 32 -> 14 * 14 * 32  
        x = torch.relu(self.conv2(x))    # 14 * 14 * 32 -> 14 * 14 * 16
        x = self.pool(x)                 # 14 * 14 * 16 -> 7 * 7 * 16  
        x = torch.relu(self.conv3(x))    # 7 * 7 * 16   -> 7 * 7 * 5
        x = self.flatten(x)              # 7 * 7 * 5     -> 1 * 49 * 5
        x = self.fc1(x)                  # 1 * 49 * 5   -> 1 x 10
        x = torch.relu(x)                # 1 * 10

        return x

for lr in learning_rates:
    for num_epochs in num_epochs_list:
        for w0 in w0_values :

            for optimizer_name, optimizer_class in optimizers.items():
                print("Optimizer ", optimizer_name)

                print("Epoch Sayisi: ", num_epochs, "  learning_rate: ", lr, "  batch_size: ", batch_size)
                print("------------------------------------------------------------")
                model = SimpleCNN()
                #model.initialize_weights(w0)  
                
                if optimizer_name == 'SGD':
                    optimizer = optimizer_class(model.parameters(), lr=lr)
                elif optimizer_name == 'SGD_Momentum':
                    optimizer = optimizer_class(model.parameters(), lr=lr, momentum=0.9)
                else:
                    optimizer = optimizer_class(model.parameters(), lr=lr)

                criterion = nn.CrossEntropyLoss() 

                losses = []
                accuracies = []
                weights = []
                weights_list = []
                initial_weights = []
                epoch_times = []

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        initial_weights.append(param.data.numpy().flatten())
                initial_weights = np.concatenate(initial_weights, axis=None)


                weights_list=initial_weights
                start_time = time.time()

                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    correct_predictions = 0
                    total_samples = 0

                    for batch_X, batch_y in trainloader: 

                        optimizer.zero_grad()

                        outputs = model(batch_X)  
    
                        loss = criterion(outputs, batch_y)
                        loss.backward()

                        optimizer.step()
                        epoch_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total_samples += batch_y.size(0)
                        correct_predictions += (predicted == batch_y).sum().item()
                    
                    weights = []
                    with torch.no_grad():
                        for name,param in model.named_parameters():
                            weights.append(param.data.numpy().flatten())
                    weights = np.concatenate(weights, axis=None)
                    weights_list=np.vstack((weights_list,weights)) 

                   
                    accuracy = correct_predictions / total_samples
                    accuracies.append(accuracy)
                    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(trainloader)}, Accuracy: {accuracy}')
                    losses.append(epoch_loss / len(trainloader))
                    epoch_times.append(time.time() - start_time)    

                end_time = time.time() 
                elapsed_time = end_time - start_time 

                
                last_weights = []
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        last_weights.append(param.data.numpy().flatten())

                
                last_weights = np.concatenate(last_weights, axis=None)
                weights_list=np.vstack((weights_list,last_weights))
                print("weights_list:")  
                print(len(weights_list))

                results[optimizer_name] = {
                    'losses': losses,
                    'accuracies': accuracies, 
                    'weights' : weights_list,
                    'elapsed_time': elapsed_time,
                    'epoch_times': epoch_times
                }
                print(f"Optimizer: {optimizer_name}, w0 = {w0}, Elapsed Time: {elapsed_time} seconds")


            print("--------- Weight Degisiyor ----yeni initial_weight: ", w0)
            
            # Sonuçların görselleştirilmesi
            plt.figure(figsize=(12, 5))

            # Loss değerlerini görselleştirme
            plt.subplot(1, 2, 1)
            for optimizer_name, result in results.items():
                plt.plot(result['losses'], label=optimizer_name)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.title(f'Loss (LR={lr}), Epochs={num_epochs}, W0={w0})')
            plt.legend()

            # Accuracy değerlerini görselleştirme
            plt.subplot(1, 2, 2)
            for optimizer_name, result in results.items():
                plt.plot(result['accuracies'], label=optimizer_name)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy (LR={lr}), Epochs={num_epochs}, W0={w0})')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'Three algorithm Epochs={num_epochs},LR={lr}, w0: {w0}.png')


            plt.show(block=False)


            dbfile = open(f'examplePickle_w0:{w0}', 'ab') 
            pickle.dump(results, dbfile)                    
            dbfile.close()

            plt.figure(figsize=(12, 5))

            # weight değerlerini görselleştirme
            plt.subplot(1, 2, 1)
            for name,result in results.items():
                total_weights_list = result['weights']
                print("TOTAL WEIGHTS")
                print(total_weights_list)
                tsne_ = TSNE(n_components=2, random_state=42,perplexity=5).fit_transform(total_weights_list)
                print("TSNE")
                print(tsne_)
                plt.scatter(tsne_[:, 0], tsne_[:, 1], label=name, alpha=0.7)
                
                for i, txt in enumerate(range(1)):
                        plt.annotate(txt, (tsne_[i, 0], tsne_[i, 1]))

            
            plt.title('t-SNE Visualization')
            plt.legend()

            plt.subplot(1, 2, 2)

            for optimizer_name, result in results.items():
                weights_list_ = result['weights']
                pca_ = PCA(n_components=2).fit_transform(weights_list_)

                pca_init,dummy = pca_[:(len(weights_list_))], pca_[(len(weights_list_)):]
                plt.scatter(pca_init[:, 0], pca_init[:, 1], label=optimizer_name, alpha=0.7)
                
                for i, txt in enumerate(range(1)):
                    plt.annotate(txt, (pca_[i, 0], pca_[i, 1]))            

            plt.title('PCA Visualization')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'PCA-TSNE_3 Epochs={num_epochs},LR={lr}, w0: {w0}.png')

            plt.show(block=False)




########################
