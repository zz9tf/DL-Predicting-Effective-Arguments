import os
import torch
from torchtext.legacy import data

#set batch size
BATCH_SIZE = 64

# print(torch.__version__)
TEXT = data.Field(tokenize='spacy', batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)

fields = [("discourse_id", None), ("essay_id", None), ('discourse_text',TEXT), ("discourse_type", None), ('discourse_effectiveness', LABEL)]

current_path = os.getcwd()
data_dir = '/../data/'
train_data_path = os.path.join(current_path + data_dir, "train.csv")
training_data=data.TabularDataset(path = train_data_path,format = 'csv', fields = fields, skip_header = True)
# This is how to view tabulardataset based on the idx:
# print(vars(training_data.examples[0]))

train_data, valid_data = training_data.split(split_ratio=0.7)

# vectors – one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes.
# Alternatively, one of or a list of available pretrained vectors: charngram.100d fasttext.en.300d
# fasttext.simple.300d glove.42B.300d glove.840B.300d glove.twitter.27B.25d glove.twitter.27B.50d
# glove.twitter.27B.100d glove.twitter.27B.200d glove.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d
# initialize glove embeddings (We can also experiment with different kinds of word embeddings in the future)
TEXT.build_vocab(train_data,min_freq=3, vectors ="glove.6B.100d")
LABEL.build_vocab(train_data)

# print(type(TEXT.vocab))
# #No. of unique tokens in text
# print("Size of TEXT vocabulary:",len(TEXT.vocab))
#
# #No. of unique tokens in label
# print("Size of LABEL vocabulary:",len(LABEL.vocab))

#Commonly used words
# print(TEXT.vocab.freqs.most_common(10))

#Word dictionary
# print(TEXT.vocab.stoi)
# print(LABEL.vocab.stoi)
#check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.discourse_text),
    sort_within_batch=True,
    device = device)

for batch in train_iterator:
    print(batch.discourse_text)
    print(batch.discourse_text[0].size()) # batch size * number of dimension
    print(batch.discourse_text[1].size()) # batch size
    print(batch.discourse_effectiveness)
    break