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

if torch.cuda.is_available():  
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


model_bag0 = pickle.load(open('1_train/model_from_bag0_bs32_e8_lr0.0001.pkl', 'rb'))
model_bag1 = pickle.load(open('1_train/model_from_bag1_bs16_e4_lr0.0005.pkl', 'rb'))
model_bag2 = pickle.load(open('1_train/model_from_bag2_bs16_e5_lr2e-05.pkl', 'rb'))
model_bag3 = pickle.load(open('1_train/model_from_bag3_bs32_e5_lr2e-05.pkl', 'rb'))
model_bag4 = pickle.load(open('1_train/model_from_bag4_bs32_e6_lr5e-05.pkl', 'rb'))


df_test=pd.read_csv("imdb_turkish/test.csv",index_col=[0],encoding="windows-1252")

comments1 = df_test.comment.values
labels1 = df_test.Label.values

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")

comments1 = comments1.tolist()
indices1=tokenizer.batch_encode_plus(comments1,max_length=128,add_special_tokens=True, return_attention_mask=True,pad_to_max_length=True,truncation=True)
input_ids1=indices1["input_ids"]
attention_masks1=indices1["attention_mask"]

prediction_inputs1= torch.tensor(input_ids1)
prediction_masks1 = torch.tensor(attention_masks1)
prediction_labels1 = torch.tensor(labels1)

# Set the batch size.  
batch_size = 32 

# Create the DataLoader.
prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1, prediction_labels1)
prediction_sampler1 = SequentialSampler(prediction_data1)
prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))

models = [model_bag0,
          model_bag2,
          model_bag3,
          model_bag4]

all_predictions = []
all_true_labels = []
model_index = 0
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
    model_index += 1
    print("----------------------------------")

pickle.dump(all_predictions, open(f'all_predictions_withoutbag1.pkl', 'wb'))
pickle.dump(all_true_labels, open(f'all_true_labels_withoutbag1.pkl', 'wb'))

