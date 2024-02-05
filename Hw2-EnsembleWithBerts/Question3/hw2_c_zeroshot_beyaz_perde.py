from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


## reference: https://joeddav.github.io/blog/2020/05/29/ZSL.html

def most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

df_test=pd.read_csv("final(2)/final/datasets/beyaz_perde/test.csv",index_col=[0],encoding="windows-1252")

comments = df_test.comment.values
test_labels = df_test.Label.values

big_bert_tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased',do_lower_case=True)
big_bert_model = AutoModel.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=2)

small_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-small-bert-uncased',do_lower_case=True)
small_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-small-bert-uncased', num_labels=2)

medium_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-medium-bert-uncased',do_lower_case=True)
medium_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-medium-bert-uncased', num_labels=2)

tiny_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-tiny-bert-uncased',do_lower_case=True)
tiny_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-tiny-bert-uncased', num_labels=2)

cosmos_bert_tokenizer = AutoTokenizer.from_pretrained('ytu-ce-cosmos/turkish-base-bert-uncased',do_lower_case=True)
cosmos_bert_model = AutoModel.from_pretrained('ytu-ce-cosmos/turkish-base-bert-uncased', num_labels=2)

all_tokenizers = [big_bert_tokenizer, small_bert_tokenizer, medium_bert_tokenizer, tiny_bert_tokenizer, cosmos_bert_tokenizer]
all_models = [big_bert_model, small_bert_model, medium_bert_model, tiny_bert_model, cosmos_bert_model]

model_names = ["big_bert_model", "small_bert_model","medium_bert_model","tiny_bert_model", "cosmos_bert_model"]
f = open("zeroshot_report_for_beyaz_perde_final.txt", "a")

all_predictions_for_ensemble = []
counter = 0
for tokenizer, model in zip(all_tokenizers, all_models):

    all_predictions = []

    for i in range(500) :
        sentence = comments[i]
        labels = ["kötü","güzel"]

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

        
        # print(closest[0])  # negatif için 1 pozitif için 0 döndürüyor
        
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