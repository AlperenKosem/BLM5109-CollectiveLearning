import torch
import numpy as np
import pandas as pd 
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from transformers import get_linear_schedule_with_warmup

import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

#### references:
# https://www.kaggle.com/code/pashupatigupta/transfer-learning-using-bert-in-depth

def most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

if torch.cuda.is_available():  
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


df_train=pd.read_csv("hw2_b/imdb_turkish/train.csv",index_col=[0],encoding="windows-1252")
df_test=pd.read_csv("hw2_b/imdb_turkish/test.csv",index_col=[0],encoding="windows-1252")


print(df_train["Label"].value_counts())
print("--Dataset is highly balanced---")


comments = df_train.comment.values
labels = df_train.Label.values

test_comments = df_test.comment.values
test_labels = df_test.Label.values

#Load BERT Turkish tokenizer
#tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-small-bert-uncased")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")


comments_list = comments.tolist()
indices=tokenizer.batch_encode_plus(comments_list, max_length=100,add_special_tokens=True, return_attention_mask=True,padding=True,truncation=True)
input_ids=indices["input_ids"]
attention_masks=indices["attention_mask"]



num_bagging_iterations = 11

bagged_train = []  
bagged_label = []
bagged_masks = []  

for i in range(num_bagging_iterations):
    X_sampled, y_sampled, mask_sampled = resample(input_ids, labels, attention_masks, random_state=(i+2)*33, replace=True)

     
    bagged_train.append(X_sampled)
    bagged_label.append(y_sampled)
    bagged_masks.append(mask_sampled)

_, test_inputs, _, test_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.2)

raw_num_ones = np.sum(labels == 1)
raw_num_zeros = np.sum(labels == 0)
ratio = raw_num_ones / raw_num_zeros if raw_num_zeros != 0 else float('inf')
print("Pos / Neg ratio Dataset : " ,  ratio)

for i in range(num_bagging_iterations):
    num_ones = np.sum(bagged_label[i] == 1)
    num_zeros = np.sum(bagged_label[i] == 0)
    ratio = num_ones / num_zeros if num_zeros != 0 else float('inf')
    print("Pos / Neg Bag ",i,": " ,  ratio)
    
print("-------------------------")


bag_counter = 0
classifiers = [] # tum classifierlari burada tutuyorum
accuracies = [] #tekil sonuçlar bu arrayde
all_predictions = [] # ensemble sonucu için de tüm predictionları burada tutup test_labels ile karşılaştıracağım.

f = open("hw2b_final_report_for_big_bert_beyaz_perde.txt", "a")

for X_sampled, y_sampled, attention_masks in zip(bagged_train, bagged_label, bagged_masks ):

    print("İşlenen Bag: ", bag_counter)

    print("Sampled Data (X) Length:", len(X_sampled))
    print("Sampled Labels (y) Length:", len(y_sampled))


    train_inputs, validation_inputaaas, train_labels, validation_labelees = train_test_split(X_sampled, y_sampled, random_state=42, test_size=0.2)

    if(bag_counter == 0 ) :
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(train_inputs, train_labels)
        y_pred = rf_classifier.predict(test_inputs)
        accuracy = accuracy_score(test_labels, y_pred)
        print(y_pred)
        print(test_labels)

        all_predictions.append(y_pred)
        classifiers.append(rf_classifier)
        accuracies.append(accuracy)

        print(f"Random Forest Sınıflandırıcısı Doğruluk Oranı: {accuracy}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- RandomForestClassifier_report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of RandomForestClassifier_report is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")


    elif(bag_counter == 1 ) :
        
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(train_inputs, train_labels)


        y_pred = dt_classifier.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"Decision Tree Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(dt_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- DecisionTreeClassifier Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of DecisionTreeClassifier is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")


    elif(bag_counter == 2) :
        svm_classifier = SVC(kernel='rbf', random_state=42)  # RBF (Radial Basis Function)
        svm_classifier.fit(train_inputs, train_labels)

        y_pred = svm_classifier.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"SVM Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(svm_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- Support Vector Machines Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of Support Vector Machines is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")


    elif(bag_counter == 3) :
        knn_classifier = KNeighborsClassifier(n_neighbors=4)  
        knn_classifier.fit(train_inputs, train_labels)

        y_pred = knn_classifier.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"K-NN Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(knn_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- KNeighborsClassifier Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of KNeighborsClassifier is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")
        
    elif(bag_counter == 4) :
        naive_bayes = MultinomialNB()
        naive_bayes.fit(train_inputs, train_labels)

        y_pred = naive_bayes.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"Naive Bayes Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(naive_bayes)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- NaiveBayes Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of NaiveBayes is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")

    elif(bag_counter == 5) :
        gbm_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        gbm_classifier.fit(train_inputs, train_labels)

        y_pred = gbm_classifier.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"GBM Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(gbm_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- Gradient Boosting Classifier Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of Gradient Boosting Classifier is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")
        
    elif(bag_counter == 6) :
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(train_inputs, train_labels)

        y_pred = dt_classifier.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"Decision Tree Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(dt_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- DecisionTreeClassifier Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of DecisionTreeClassifier is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")


    elif(bag_counter == 7) :
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(train_inputs, train_labels)
        y_pred = dt_classifier.predict(test_inputs)
        accuracy = accuracy_score(test_labels, y_pred)
        print(f"Decision Tree Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(dt_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        
        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- DecisionTreeClassifier Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of DecisionTreeClassifier is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")


    elif(bag_counter == 8) :
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(train_inputs, train_labels)
        y_pred = rf_classifier.predict(test_inputs)
        accuracy = accuracy_score(test_labels, y_pred)

        classifiers.append(rf_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)
        
        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        print(f"Random Forest Sınıflandırıcısı Doğruluk Oranı: {accuracy}")

        f.write(f" -- RandomForestClassifier_report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of RandomForestClassifier_report is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")


    elif(bag_counter == 9) :
        svm_classifier = SVC(kernel='rbf', random_state=42)  # RBF (Radial Basis Function)
        svm_classifier.fit(train_inputs, train_labels)

        y_pred = svm_classifier.predict(test_inputs)

        accuracy = accuracy_score(test_labels, y_pred)
        print(f"SVM Modeli Doğruluk Oranı: {accuracy}")
        classifiers.append(svm_classifier)
        accuracies.append(accuracy)
        all_predictions.append(y_pred)

        print("\nSınıflandırma Raporu:")
        print(classification_report(test_labels, y_pred))

        f.write(f" -- Support Vector Machines Report ---- \n")
        f.write(classification_report(test_labels, y_pred))
        f.write(f"Accuracy of Support Vector Machines is: ")
        f.write(str(accuracy))
        f.write(" \n ")
        f.write(" --------------------------------------\n ")

    elif(bag_counter == 10) : # bu sadece ensemble için oluşturuluyor
        
        # basta ensemblei kütüphane kullanarak oluşturmuştum onun için bu değişken var sonradan bunu kullanmaktan vazgeçtim
        classifiers___ = [
            ('RandomForestClassifier1', classifiers[0]),
            ('DecisionTreeClassifier1',  classifiers[1]),
            ('SVM',  classifiers[2]),
            ('Naive Bayes',  classifiers[3]),
            ('K-Nearest Neighbors',  classifiers[4]),
            ('GradientBoostingClassifier',  classifiers[5]),
            ('DecisionTreeClassifier2',  classifiers[6]),
            ('DecisionTreeClassifier3',  classifiers[7]),
            ('RandomForestClassifier2',  classifiers[8]),
            ('SVM2',  classifiers[9]),

        ]

        result = np.apply_along_axis(most_common, axis=0, arr=all_predictions) # sütun olarak en çok hangi karar denmiş bul ve tek boyuta indir

        print("-----Result: ", result)
        print("-----Test Labels: ", test_labels)

        ensemble_predictions_1d = np.array(result).flatten()
        ensemble_true_labels_1d = np.array(test_labels).flatten() # hepsinde aynı test datayı kullandığımız için her 5 model için de true label aynı 

        print("Accuracy of BERT is:",accuracy_score(ensemble_true_labels_1d, ensemble_predictions_1d))

        print(classification_report(ensemble_true_labels_1d, ensemble_predictions_1d))

        f.write(classification_report(ensemble_true_labels_1d, ensemble_predictions_1d))
        f.write("Accuracy of Ensemble is: ")
        f.write(str(accuracy_score(ensemble_true_labels_1d, ensemble_predictions_1d)))

    print("-------------------------")

    bag_counter += 1


pickle.dump(classifiers, open(f'hw2_b/all_classifiers.pkl', 'wb'))
