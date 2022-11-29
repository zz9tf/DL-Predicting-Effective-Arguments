import pandas as pd
import numpy as np
import train
import torch
import torch.nn as nn
import torch.optim as optim
import preprocessing as preprocess
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
EMBEDDING_DIM = 100
NUM_CLASSES = 3
data = pd.read_csv('../data/train.csv')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["discourse_text"])
dict = {'Adequate': 0,'Effective':1,'Ineffective':2 }
Y = list(data["discourse_effectiveness"])
y=list()
for i in Y:
    if i=='Adequate':
        y.append(0)
    elif i=='Effective':
        y.append(1)
    else:
        y.append(2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()

# ----- 3. Predict -----#
# Load test data
test_data = pd.read_csv("../data/test.csv")
X_test = list(test_data["discourse_text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "output/checkpoint-50000"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
# # settings
# settings_path = 'config.yml'
# settings = yaml.safe_load(open(settings_path, "r"))

# train_iterator, valid_iterator, test_iterator, TEXT, LABEL = preprocess.load_data(
#     BATCH_SIZE=settings["general"]["BATCH_SIZE"],
#     split_ratio=settings["general"]["split_ratio"],
#     data_information=settings["general"]["data_information"])

# embedding_dim = EMBEDDING_DIM
# num_classes = NUM_CLASSES
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=float(settings["training"]["weight_decay"]))
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
# train_loss, train_accuracy, valid_loss, valid_accuracy = train.model_train(net=model,
#                                                                            train_iterator=train_iterator,
#                                                                            valid_iterator=valid_iterator,
#                                                                            epoch_num=settings["training"]["EPOCH_NUM"],
#                                                                            criterion=criterion,
#                                                                            optimizer=optimizer,
#                                                                            scheduler=scheduler,
#                                                                            device=device)