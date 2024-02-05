import torch
import numpy as np
import pandas as pd 
from transformers import AutoTokenizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords

import pickle
from sklearn.metrics import accuracy_score


def most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

def clean_text(text):

    unwanted_pattern = r'[!.\n,:“”,?@#"]'
    regex = re.compile(unwanted_pattern)
    cleaned_text = regex.sub(" ", text)
    
    return cleaned_text


if torch.cuda.is_available():  
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


model_bag0 = pickle.load(open('model_from_bag0_bs32_e8_lr5e-05.pkl', 'rb'))
model_bag1 = pickle.load(open('model_from_bag1_bs32_e5_lr0.0005.pkl', 'rb'))
model_bag2 = pickle.load(open('model_from_bag2_bs16_e6_lr2e-05.pkl', 'rb'))
model_bag3 = pickle.load(open('model_from_bag3_bs32_e3_lr2e-05.pkl', 'rb'))
model_bag4 = pickle.load(open('model_from_bag4_bs16_e2_lr1e-05.pkl', 'rb'))


df_test=pd.read_csv("hw2_a/magaza_yorum/magaza_yorumlari_duygu_analizi_test.csv", encoding = "utf-16")
print(df_test.head())

print(df_test["Durum"].value_counts())

df_test['Durum'] = df_test.loc[:, 'Durum'].map({'Olumlu' : 0, 'Olumsuz' : 1, 'Tarafsız' : 2 })
print(df_test.head())

df_test['new_text'] = df_test['Görüş'].astype(str).apply(clean_text)
df_test['new_text'] = df_test['new_text'].str.replace("[\d]", "")
df_test['new_text'] = df_test['new_text'].str.lower()

ineffective = stopwords.words('turkish')

df_test['new_text'] = df_test['new_text'].apply(lambda x: " ".join(x for x in x.split() if x not in ineffective))
print(df_test.head())


comments = df_test.new_text.values
labels = df_test.Durum.values

print(len(comments))
print(len(labels))

#Load BERT Turkish tokenizer
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-small-bert-uncased")


comments1 = comments.tolist()
indices1=tokenizer.batch_encode_plus(comments1,max_length=100,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)
input_ids1=indices1["input_ids"]
attention_masks1=indices1["attention_mask"]

prediction_inputs1= torch.tensor(input_ids1)
prediction_masks1 = torch.tensor(attention_masks1)
prediction_labels1 = torch.tensor(labels)

# Set the batch size.  
batch_size = 32 

# Create the DataLoader.
prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1, prediction_labels1)
prediction_sampler1 = SequentialSampler(prediction_data1)
prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))


models = [model_bag0,
          model_bag1,
          model_bag2,
          model_bag3,
          model_bag4
          ]

all_predictions = []
all_true_labels = []
ensemble_predictions = []
ensemble_true_labels = []
model_index = 0

f = open("report_for_small_bert_for_magaza_yorum_withoutbag1.txt", "a")

for model in models :
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader1:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids1, b_input_mask1, b_labels1 = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs1 = model(b_input_ids1, token_type_ids=None, 
                            attention_mask=b_input_mask1)

        logits1 = outputs1[0]

        # Move logits and labels to CPU
        logits1 = logits1.detach().cpu().numpy()
        label_ids1 = b_labels1.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits1)
        true_labels.append(label_ids1)

    print('    DONE.')
    print(f'MODEL : {model_index} ------ :')

    all_predictions.append(predictions)
    all_true_labels.append(true_labels)

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    print("Accuracy of BERT is:",accuracy_score(flat_true_labels, flat_predictions))

    print(classification_report(flat_true_labels, flat_predictions))
    
    
    ensemble_predictions.append(flat_predictions)
    ensemble_true_labels.append(flat_true_labels)
    ####Dosyaya Yazma 

    f.write(f" -- Bag:{model_index} classification_report ---- \n")
    f.write(classification_report(flat_true_labels, flat_predictions))
    f.write(f"Accuracy of Bag {model_index} is: ")
    f.write(str(accuracy_score(flat_true_labels, flat_predictions)))
    f.write(" \n ")
    f.write(" --------------------------------------\n ")


    model_index += 1
    print("----------------------------------")

f.write(" -----------Ensemble Report ----------\n ")

result = np.apply_along_axis(most_common, axis=0, arr=ensemble_predictions)

ensemble_predictions_1d = np.array(result).flatten()
ensemble_true_labels_1d = np.array(ensemble_true_labels[0]).flatten() # hepsinde aynı test datayı kullandığımız için her 5 model için de true label aynı o yüzden sadece ilkini vermek yeterli


print(len(ensemble_predictions_1d)) # 2666 * 5
print(len(ensemble_true_labels_1d))

print("Accuracy of BERT is:",accuracy_score(ensemble_true_labels_1d, ensemble_predictions_1d))

print(classification_report(ensemble_true_labels_1d, ensemble_predictions_1d))

f.write(classification_report(ensemble_true_labels_1d, ensemble_predictions_1d))
f.write("Accuracy of Ensemble is: ")
f.write(str(accuracy_score(ensemble_true_labels_1d, ensemble_predictions_1d)))


f.close()


pickle.dump(all_predictions, open(f'all_predictions_magaza_yorum.pkl', 'wb'))
pickle.dump(all_true_labels, open(f'all_true_labels_magaza_yorum.pkl', 'wb'))
