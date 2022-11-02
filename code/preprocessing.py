import os
import torch
from torchtext.legacy import data


def load_data(BATCH_SIZE=10,
              split_ratio=0.7,
              vectors="glove.6B.100d",
              data_information=False
              ):
    """
    This function loads our datasets which are wrapped by Torchtext library
    :param BATCH_SIZE: batch size (default: 10)
    :param split_ratio: Ratio between training and validation datasets (default: 0.7)
    :param vectors: pre-trained word embeddings we want to use (default: "glove.6B.100d")
    :param data_information: True if you want to view information of training datasets (default: False)
    :return: train_iterator, valid_iterator, test_iterator, TEXT, LABEL
    """
    TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    train_fields = [("discourse_id", None), ("essay_id", None), ('discourse_text', TEXT), ("discourse_type", None),
                    ('discourse_effectiveness', LABEL)]
    test_fields = [("discourse_id", None), ("essay_id", None), ('discourse_text', TEXT), ("discourse_type", None)]

    current_path = os.getcwd()
    data_dir = '/../data/'
    train_data_path = os.path.join(current_path + data_dir, "train.csv")
    training_data = data.TabularDataset(path=train_data_path, format='csv', fields=train_fields, skip_header=True)

    test_data_path = os.path.join(current_path + data_dir, "test.csv")
    test_data = data.TabularDataset(path=test_data_path, format='csv', fields=test_fields, skip_header=True)

    train_data, valid_data = training_data.split(split_ratio=split_ratio)
    TEXT.build_vocab(train_data, min_freq=3, vectors=vectors)
    LABEL.build_vocab(train_data)

    if data_information:
        print("Training Data information")
        print(type(TEXT.vocab))
        # No. of unique tokens in text
        print("Size of TEXT vocabulary:", len(TEXT.vocab))

        # No. of unique tokens in label
        print("Size of LABEL vocabulary:", len(LABEL.vocab))

        # Commonly used words
        print("Ten most commly used words are", TEXT.vocab.freqs.most_common(10))

        # Word dictionary
        # print(TEXT.vocab.stoi)
        # print(LABEL.vocab.stoi)
    # check whether cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load an iterator
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.discourse_text),
        sort_within_batch=True,
        device=device)
    test_iterator = data.BucketIterator(
        test_data,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.discourse_text),
        sort_within_batch=True,
        device=device)

    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL


if __name__ == "__main__":
    train_iterator, valid_iterator, test_iterator, TEXT, LABEL = load_data(BATCH_SIZE=1)
    for batch in train_iterator:
        # print(batch.discourse_text)
        print(batch.discourse_text[0].size())  # batch size * number of dimension
        print(batch.discourse_text[1].size())  # batch size
        # print(batch.discourse_effectiveness)
        break
    print("=" * 10)
    for batch in test_iterator:
        # print(batch.discourse_text)
        print(batch.discourse_text[0].size())  # batch size * number of dimension
        print(batch.discourse_text[1].size())  # batch size
        # print(batch.discourse_effectiveness)
        break

    for batch in valid_iterator:
        # print(batch.discourse_text)
        print(batch.discourse_text[0].size())  # batch size * number of dimension
        print(batch.discourse_text[1].size())  # batch size
        # print(batch.discourse_effectiveness)
        break


    print(TEXT.vocab.vectors.size())
