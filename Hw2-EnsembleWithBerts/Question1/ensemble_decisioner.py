import torch
import numpy as np
import pandas as pd 
from transformers import AutoTokenizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

import pickle
from sklearn.metrics import accuracy_score

def most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]



all_predictions = pickle.load(open('hw2_a/small_bert_beyaz_perde/all_predictions_small_bert.pkl', 'rb'))
all_true_labels = pickle.load(open('hw2_a/small_bert_beyaz_perde/all_true_labels_small_bert.pkl', 'rb'))

ensemble_predictions = []
ensemble_true_labels = []


f = open("report_for_small_bert_for_beyaz_perde.txt", "a")
index = 0
for prediction, true_label in zip(all_predictions, all_true_labels):
    flat_predictions = [item for sublist in prediction for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_label for item in sublist]
    ensemble_predictions.append(flat_predictions)
    ensemble_true_labels.append(flat_true_labels)

    f.write(f" -- Bag:{index} classification_report ---- \n")
    f.write(classification_report(flat_true_labels, flat_predictions))
    f.write(f"Accuracy of Bag {index} is: ")
    f.write(str(accuracy_score(flat_true_labels, flat_predictions)))
    f.write(" \n ")
    index += 1
    f.write(" --------------------------------------\n ")



result = np.apply_along_axis(most_common, axis=0, arr=ensemble_predictions)


f.write(" -----------Ensemble Report ----------\n ")

ensemble_predictions_1d = np.array(result).flatten()
ensemble_true_labels_1d = np.array(ensemble_true_labels[0]).flatten() # hepsinde aynı test datayı kullandığımız için her 5 model için de true label aynı


print(len(ensemble_predictions_1d))
print(len(ensemble_true_labels_1d))

print("Accuracy of BERT is:",accuracy_score(ensemble_true_labels_1d, ensemble_predictions_1d))

print(classification_report(ensemble_true_labels_1d, ensemble_predictions_1d))

f.write(classification_report(ensemble_true_labels_1d, ensemble_predictions_1d))
f.write("Accuracy of Ensemble is: ")
f.write(str(accuracy_score(ensemble_true_labels_1d, ensemble_predictions_1d)))


f.close()