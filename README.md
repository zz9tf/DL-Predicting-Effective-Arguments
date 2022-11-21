# DL-project

### Predicting Effective Arguments

A model which classifies argumentative elements such as Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, and Concluding Statement as "effective," "adequate," or "ineffective." based on essays written by U.S. students in grades 6-12.

### Installation and Dependencies:
We recommend using conda environment to install dependencies of this library first. Please install (or load) conda and then proceed with the following commands:
```
conda create -n dl_project python=3.9
conda activate dl_project
conda install pytorch torchvision torchaudio -c pytorch
conda install -c pytorch torchtext
conda install matplotlib
pip install pyyaml
```

Sometimes, there might be some errors when using torchtext such as
```
ModuleNotFoundError: No module named 'torchtext.legacy'
```

In this case, try downgrade your torchtext 
```
pip install torchtext==0.10.0
```
Next, please install spacy since it is our default tokenizer
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
or, if you are using ARM/M1, run this:
```
pip install -U pip setuptools wheel
pip install -U 'spacy[apple]'
python -m spacy download en_core_web_sm
```

### Code Hierarchy Table

- Inside the code folder

| File                                      | Description                                                                                             |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [preprocessing.py](code/preprocessing.py) | Splits and wraps datasets into data iterators using Torchtext                                           |
| [rnn.py](code/rnn.py)                     | Includes a RNN model and a fine tune version which trains/evaluates this RNN model                      |
 | [models.py](code/models.py)               | Defines the RNN, LSTM, and GRU                                                                          |                         
| [config.yml](code/config.yml)             | Sets the hyperparameter for loading data, initializing models, and training.                            |
| [train.py](code/train.py)                 | Defines functions that train the model, plot loss/accuracy for train/valid datasets, and make inference |
| [run.py](code/run.py)                     | Trains the model, plot loss and accuracy                                                                |

### Model's accuracy
| Model | Bidirectional | Last Hidden | Accuracy/Loss on train | Accuracy/Loss on validation | 
|-------|---------------|-------------|------------------------|-----------------------------|
| RNN   | True          | True        | 68.78% / 0.817         | 64.8% / 0.873               |
| RNN   | True          | False       | ?/?                    | ?/?                         |
| LSTM  | True          | True        | 73.44% / 0.816         | 63.49% / 0.912              |
| LSTM  | True          | False       | ?/?                    | ?/?                         |
| GRU   | True          | True        | 76.10% / 0.79          | 63.40% / 0.914              |
| GRU   | True          | False       | ?/?                    | ?/?                         |

### Model structure

| Model layers           | input                         | output                                                                                           | notes                                                                                                         |
|------------------------|-------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Embedding layer        | (batch_size, sequence_length) | (batch_size, sequence_length, embedding_size)                                                    | Encode words as word embedding by using pretrained text vectors                                               |
| RNN layer              | PACKEDSEQUENCE                | (sequence_length, batch_size, (D * hidden_size)) and ((D * num_layers), batch_size, hidden_size) | Count final forward and backward hidden states as the RNN output, or the average pooling of all hidden states |
| Fully connection layer | (batch_size, D * hidden_size) | (batch_size, 3)                                                                                  | Combine rnn result and get output shape match the output size                                                 |
| softmax layer          | (batch_size, 3)               | (batch_size, 3)                                                                                  | Softmax output for our text classification problem                                                            |

## RNN model

### Introduction
In this section, to get the classification result, we used many-to-one RNN model the processing the whole sentence and produced the result.

![image](https://user-images.githubusercontent.com/77183284/198885542-63c77159-b458-49fd-9b5f-6036082efebc.png)

- Model pramater

<img width="465" alt="image" src="https://user-images.githubusercontent.com/77183284/200183904-bda81a48-41c0-49ff-9185-fe0d3752819c.png">




