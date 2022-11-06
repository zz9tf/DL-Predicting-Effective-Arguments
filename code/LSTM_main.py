import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import LSTM_util as U
import preprocessing as preprocess
from sklearn.metrics import accuracy_score
import numpy as np

torch.manual_seed(1)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
train_iterator, valid_iterator, test_iterator, TEXT, LABEL = preprocess.load_data(BATCH_SIZE=1)
model = U.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(TEXT.vocab), len(LABEL.vocab))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
def train(epoch,model,train_iterator,valid_iterator):
    for epoch in range(epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        for batch in train_iterator:
            model.zero_grad()
            X = batch.discourse_text[0]
            Y = batch.discourse_effectiveness
            print("X", X)
            print("effectiveness",Y)
            tag_scores = model(X)
            loss = loss_function(tag_scores, Y.to(torch.int64))
            loss.backward()
            optimizer.step()
        test(valid_iterator)

def test(valid_iterator):
    losses = []
    for batch in valid_iterator:
        X = batch.discourse_text[0]
        Y = batch.discourse_effectiveness
        with torch.no_grad():
            tag_scores = model(X)
            loss = loss_function(tag_scores, Y.to(torch.int64))
            losses.append(loss.item())
        
        softmax = torch.exp(tag_scores)
        prob = list(softmax.numpy())
        y_test_predictions = np.argmax(prob, axis=1)
        
        test_data_y = Y.numpy()
        # accuracy on test set
        accuracy = accuracy_score(test_data_y.T, y_test_predictions)
        print("Test_accuracy:", accuracy)

train(100,model,train_iterator,valid_iterator)
