import torch
import torch.nn as nn
import torch.optim as optim
import yaml
# from pytorch_forecasting.optim import Ranger

import models as U
import preprocessing as preprocess
import train

# DO NOT Modify
EMBEDDING_DIM = 100
NUM_CLASSES = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# settings
settings_path = 'config.yml'
settings = yaml.safe_load(open(settings_path, "r"))

model_name = settings["general"]["model"].upper()  # ("RNN", "LSTM", "GRU")

train_iterator, valid_iterator, test_iterator, TEXT, LABEL = preprocess.load_data(
    BATCH_SIZE=settings["general"]["BATCH_SIZE"],
    split_ratio=settings["general"]["split_ratio"],
    data_information=settings["general"]["data_information"])
# define hyperparameters
embedding_dim = EMBEDDING_DIM
num_classes = NUM_CLASSES

# instantiate the model
if model_name == "RNN":
    print("Running model", model_name)
    model = U.RNN(TEXT=TEXT,
                  embedding_size=embedding_dim,
                  hidden_size=settings[model_name]["hidden_size"],
                  num_classes=num_classes,
                  layers=settings[model_name]["number_of_layers"],
                  dropout=settings[model_name]["dropout"],
                  device=device,
                  bidirectional=settings[model_name]["bidirectional"],
                  last_hidden=settings[model_name]["last_hidden"])
elif model_name == "LSTM":
    print("Running model", model_name)
    model = U.LSTM(TEXT=TEXT,
                   embedding_size=embedding_dim,
                   hidden_size=settings[model_name]["hidden_size"],
                   num_classes=num_classes,
                   layers=settings[model_name]["number_of_layers"],
                   dropout=settings[model_name]["dropout"],
                   device=device,
                   bidirectional=settings[model_name]["bidirectional"],
                   last_hidden=settings[model_name]["last_hidden"])
else:
    print("Running model GRU")
    model = U.GRU(TEXT=TEXT,
                  embedding_size=embedding_dim,
                  hidden_size=settings[model_name]["hidden_size"],
                  num_classes=num_classes,
                  layers=settings[model_name]["number_of_layers"],
                  dropout=settings[model_name]["dropout"],
                  device=device,
                  bidirectional=settings[model_name]["bidirectional"],
                  last_hidden=settings[model_name]["last_hidden"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=float(settings["training"]["weight_decay"]))
# optimizer = Ranger(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=float(settings["training"]["weight_decay"]))
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
train_loss, train_accuracy, valid_loss, valid_accuracy = train.model_train(net=model,
                                                                           train_iterator=train_iterator,
                                                                           valid_iterator=valid_iterator,
                                                                           epoch_num=settings["training"]["EPOCH_NUM"],
                                                                           criterion=criterion,
                                                                           optimizer=optimizer,
                                                                           scheduler=scheduler,
                                                                           device=device)

train.plot_loss(train_loss=train_loss, valid_loss=valid_loss, model_name=model_name)
train.plot_accuracy(train_accuracy=train_accuracy, valid_accuracy=valid_accuracy, model_name=model_name)
