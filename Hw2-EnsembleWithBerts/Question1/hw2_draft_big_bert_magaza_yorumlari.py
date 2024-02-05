import torch
import numpy as np
import pandas as pd 
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.utils import resample
import re
import nltk
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AdamW, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

import random
import time
import datetime
import pickle

#### references:
# https://www.kaggle.com/code/pashupatigupta/transfer-learning-using-bert-in-depth
# https://www.kaggle.com/code/ozcan15/turkish-sentiment-analysis-with-berturk-93-acc#2)-BERTURK-Model-Fine-Tuning
# https://towardsdatascience.com/lit-bert-nlp-transfer-learning-in-3-steps-272a866570db

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def plot_sentence_embeddings_length(text_list, tokenizer): # reference:
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))
    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tokenized_texts_len, bins=40)
    ax.set_xlabel("Length of Comment Embeddings")
    ax.set_ylabel("Number of Comments")
    plt.show()

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


df_train=pd.read_csv("hw2_a/magaza_yorum/magaza_yorumlari_duygu_analizi_train.csv", encoding = "utf-16")
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
labels = df_train.Durum.values

print(len(comments))
print(len(labels))

#Load BERT Turkish tokenizer
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-small-bert-uncased")

print((comments[0]))
plot_sentence_embeddings_length(comments,tokenizer) 

num_bagging_iterations = 5

bagged_train = []  # Bagged özellikler
bagged_label = []  # Bagged etiketler

bag_counter = 0


for i in range(bag_counter,num_bagging_iterations):
    X_sampled, y_sampled = resample(comments, labels, random_state=(i+2)*33, replace=True) 
    bagged_train.append(X_sampled)
    bagged_label.append(y_sampled)

print(len(bagged_train))


for X_sampled, y_sampled in zip(bagged_train, bagged_label):

    print("İşlenen Bag: ", bag_counter)

    print("Sampled Data (X) Length:", len(X_sampled))
    print("Sampled Labels (y) Length:", len(y_sampled))
    print("-------------------------")

    comments = X_sampled.tolist()
    labels = y_sampled
    if(bag_counter == 0):
        batch_size = 32
        lr = 5e-5
        epochs = 8

    elif(bag_counter == 1) :
        batch_size = 32
        lr = 0.0005
        epochs = 5

    elif(bag_counter == 2) :
        batch_size = 16
        lr = lr = 2e-5
        epochs = 6

    elif(bag_counter == 3) :
        batch_size = 32
        lr =  2e-5
        epochs = 3

    elif(bag_counter == 4) :
        batch_size = 16
        lr =  1e-5
        epochs = 2

    else :
        batch_size = 16
        lr =  2e-5
        epochs = 6


    indices=tokenizer.batch_encode_plus(comments,max_length=100,add_special_tokens=True, return_attention_mask=True,padding=True, truncation=True)
    input_ids=indices["input_ids"]
    attention_masks=indices["attention_mask"]

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.2)

    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=42, test_size=0.2)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    train_masks = torch.tensor(train_masks, dtype=torch.long)
    validation_masks = torch.tensor(validation_masks, dtype=torch.long)


    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    config = AutoConfig.from_pretrained(
            "ytu-ce-cosmos/turkish-small-bert-uncased",num_labels=3)
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = AutoModelForSequenceClassification.from_pretrained(
        "ytu-ce-cosmos/turkish-small-bert-uncased",config=config)

    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = lr , # args.learning_rate - default is 5e-5
                    betas=[0.9,0.999],
                    eps = 1e-6 # args.adam_epsilon  - default is 1e-8.
                    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)




    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    acc_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 30 batches.
            if step % 30 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        acc_values.append(eval_accuracy/nb_eval_steps)

    print("")
    print("Training complete!")


    pickle.dump(model, open(f'model_from_bag{bag_counter}_bs{batch_size}_e{epochs}_lr{lr}.pkl', 'wb'))

    # Sonuçları görselleştirme
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    plt.plot(loss_values, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title(f'Loss AdamWOptimizer (LR={lr}, Epochs={epochs})')
    plt.legend()

    # Accuracy değerlerini görselleştirme
    plt.subplot(1, 2, 2)
    plt.plot(acc_values, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy AdamWOptimizer (LR={lr}), Epochs={epochs})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'Adamw Optimizer bag{bag_counter} Epochs={epochs},LR={lr},bs{batch_size} .png')
    plt.show(block=False)
    bag_counter += 1