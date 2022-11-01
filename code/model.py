import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocessing as preprocess

#set batch size
BATCH_SIZE = 10
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

train_iterator, valid_iterator, test_iterator, TEXT, LABEL = preprocess.load_data(BATCH_SIZE=30)