import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocessing as preprocess
from rnn import RNN
from sklearn.metrics import accuracy_score

def one_hot(Y, model):
    return F.one_hot(Y, num_classes=model.output_size).to(torch.float32)

def CalcValLossAndAccuracy(model, loss_fn, valid_iterator):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for batch in valid_iterator:
            X = batch.discourse_text[0]
            Y = batch.discourse_effectiveness.to(torch.int64)
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(Y_pred.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)
        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))

def train(model, train_iterator, valid_iterator, epochs=30):
    losses = []
    for i in range(1, epochs + 1):
        epoch_loss = []
        for batch in train_iterator:
            X = batch.discourse_text[0]
            Y = batch.discourse_effectiveness.to(torch.int64)
            Y_preds = model(X)
            loss = loss_fn(Y_preds, Y)
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {}/{}".format(i, epochs))
        print("Train Loss : {:.3f}".format(sum(epoch_loss)/len(epoch_loss)))

        losses.append(sum(epoch_loss)/len(epoch_loss))
        CalcValLossAndAccuracy(model, loss_fn, valid_iterator)
    return losses

# set batch size
BATCH_SIZE = 30
# Ratio between train datasets and validation datasets.
split_ratio = 0.7
# vectors – one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes.
# Alternatively, one of or a list of available pretrained vectors: charngram.100d fasttext.en.300d
# fasttext.simple.300d glove.42B.300d glove.840B.300d glove.twitter.27B.25d glove.twitter.27B.50d
# glove.twitter.27B.100d glove.twitter.27B.200d glove.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d
# initialize glove embeddings (We can also experiment with different kinds of word embeddings in the future)
vectors = "glove.6B.100d"
# Ture if you want to view size of TEXT vocabulary, size of Label Vocabulary, and top 10 commonly used words
data_information = False

train_iterator, valid_iterator, test_iterator, TEXT, LABEL = preprocess.load_data(BATCH_SIZE=BATCH_SIZE)

vocab = TEXT
embed_len = 4
hidden_size = 128
target_classes = LABEL

model_configs = {
    'vocab': vocab,
    'embed_len': embed_len,
    'hidden_size': hidden_size,
    'target_classes': target_classes
}
model = RNN(model_configs=model_configs)
learning_rate = 0.001
loss_fn = model.get_loss_fn()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for layer in model.children():
    print("Layer : {}".format(layer))
    print("Parameters : ")
    for param in layer.parameters():
        print(param.shape)
    print()

# for train_batch in train_iterator:
#     print(train_batch.discourse_text[0])
#     print(train_batch.discourse_effectiveness)
#     input()

train(model, train_iterator, valid_iterator)


# from sklearn.metrics import confusion_matrix
# import scikitplot as skplt
# import matplotlib.pyplot as plt
# import numpy as np
#
# skplt.metrics.plot_confusion_matrix([target_classes[i] for i in Y_actual], [target_classes[i] for i in Y_preds],
#                                     normalize=True,
#                                     title="Confusion Matrix",
#                                     cmap="Purples",
#                                     hide_zeros=True,
#                                     figsize=(5,5)
#                                     );
# plt.xticks(rotation=90);


