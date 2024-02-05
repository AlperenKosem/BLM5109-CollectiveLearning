from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import torch
import numpy as np
import pandas as pd 
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.utils import resample
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


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


df_train=pd.read_csv("final(2)/final/datasets/magaza_yorumlari/magaza_yorumlari_duygu_analizi.csv", encoding = "utf-16")
print(df_train.head())

print(df_train["Durum"].value_counts())

df_train['Durum'] = df_train.loc[:, 'Durum'].map({'Olumlu' : 0, 'Olumsuz' : 1, 'Tarafsız' : 2 })
print(df_train.head())

df_train['new_text'] = df_train['Görüş'].astype(str).apply(clean_text)
df_train['new_text'] = df_train['new_text'].str.replace("[\d]", "")
df_train['new_text'] = df_train['new_text'].str.lower()

ineffective = stopwords.words('turkish')

df_train['new_text'] = df_train['new_text'].apply(lambda x: " ".join(x for x in x.split() if x not in ineffective))
print(df_train.head())


comments = df_train.new_text.values
test_labels = df_train.Durum.values
big_bert_tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased',do_lower_case=True)
big_bert_model = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=3)

small_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-small-bert-uncased',do_lower_case=True)
small_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-small-bert-uncased', num_labels=3)

medium_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-medium-bert-uncased',do_lower_case=True)
medium_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-medium-bert-uncased', num_labels=3)

tiny_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-tiny-bert-uncased',do_lower_case=True)
tiny_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-tiny-bert-uncased', num_labels=3)

cosmos_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-base-bert-uncased',do_lower_case=True)
cosmos_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-base-bert-uncased', num_labels=3)

all_tokenizers = [big_bert_tokenizer, small_bert_tokenizer, medium_bert_tokenizer, tiny_bert_tokenizer, cosmos_bert_tokenizer]
all_models = [big_bert_model, small_bert_model, medium_bert_model, tiny_bert_model, cosmos_bert_model]

model_names = ["big_bert_model", "small_bert_model","medium_bert_model","tiny_bert_model", "cosmos_bert_model"]
f = open("zeroshot_report_for_magaza_yorum.txt", "a")

all_predictions_for_ensemble = []
counter = 0
for tokenizer, model in zip(all_tokenizers, all_models):

    all_predictions = []

    for i in range(500) :
        sentence = comments[i]
        labels = ["Olumlu","Olumsuz", "Tarafsız"]

        inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                            return_tensors='pt',
                                            padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = model(input_ids, attention_mask=attention_mask)[0]
        sentence_rep = output[:1].mean(dim=1)
        label_reps = output[1:].mean(dim=1)


        similarities = F.cosine_similarity(sentence_rep, label_reps)
        closest = similarities.argsort(descending=True)

                
        all_predictions.append(closest[0].item()) 

    print("Accuracy of BERT is:",accuracy_score(test_labels[0:len(all_predictions)], all_predictions))

    print(classification_report(test_labels[0:len(all_predictions)], all_predictions))

    ####Dosyaya Yazma 

    f.write(f" -- Model:{model_names[counter]} classification_report ---- \n")
    f.write(classification_report(test_labels[0:len(all_predictions)], all_predictions))
    f.write(f"Accuracy of Model {model_names[counter]} is: ")
    f.write(str(accuracy_score(test_labels[0:len(all_predictions)], all_predictions)))
    f.write(" \n ")
    f.write(" --------------------------------------\n ")


    all_predictions_for_ensemble.append(all_predictions)
    counter += 1


print("---Predictions :", all_predictions_for_ensemble)

result = np.apply_along_axis(most_common, axis=0, arr=all_predictions_for_ensemble)
print("-----------",result)

    ####Dosyaya Yazma 

f.write(f" -- Ensemble classification_report ---- \n")
f.write(classification_report(test_labels[0:len(result)], result))
f.write(f"Accuracy of Ensemble Model is: ")
f.write(str(accuracy_score(test_labels[0:len(result)], result)))
f.write(" \n ")
f.write(" --------------------------------------\n ")