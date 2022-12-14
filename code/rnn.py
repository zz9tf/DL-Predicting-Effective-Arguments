import preprocessing as preprocess
import time
from sklearn.metrics import accuracy_score
from functools import reduce
import torch
from torch import nn
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


class RNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RNN, self).__init__()
        model_configs = kwargs.pop('model_configs')

        # init model attributions
        self.input_size = model_configs['embed_len']
        self.hidden_size = model_configs['hidden_size']
        self.output_size = len(model_configs['target_classes'].vocab)
        self.num_output_classes = len(model_configs['vocab'].vocab)

        # init layers
        self.embedding_layer = nn.Embedding(num_embeddings=self.num_output_classes, embedding_dim=self.input_size)
        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, batch_x):
        # init hidden input
        hidden_input = torch.zeros(batch_x.shape[0], self.hidden_size).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(hidden_input)

        for word_id in range(batch_x.shape[-1]):
            embeddings = self.embedding_layer(batch_x[:, word_id])
            combined_input = torch.cat((embeddings, hidden_input), 1)
            hidden_input = self.i2h(combined_input)
        output = self.i2o(combined_input)
        output = self.softmax(output)

        return output

    def get_loss_fn(self):
        # return nn.BCELoss()
        return nn.NLLLoss()

    def get_optimizer(self, params=None, lr=1e-3):
        assert params != None
        return torch.optim.SGD(params, lr=lr)


def secondsToStr(t):
    return '%d:%02d:%02d.%03d' % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])


class Fine_tune_rnn():
    def __init__(self):
        # set batch size
        BATCH_SIZE = 30
        self.train_iterator, self.valid_iterator, \
        self.test_iterator, TEXT, LABEL = preprocess.load_data(BATCH_SIZE=BATCH_SIZE)

        self.vocab = TEXT
        embed_len = 4
        hidden_size = 128
        self.target_classes = LABEL

        model_configs = {
            'vocab': self.vocab,
            'embed_len': embed_len,
            'hidden_size': hidden_size,
            'target_classes': self.target_classes
        }
        self.model = RNN(model_configs=model_configs).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        learning_rate = 0.0005
        self.loss_fn = self.model.get_loss_fn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.show_rnn()

    def show_rnn(self):
        for layer in self.model.children():
            print("Layer : {}".format(layer))
            print("Parameters : ")
            for param in layer.parameters():
                print(param.shape)
            print()

    def train(self, epochs=1):
        losses = []
        for i in range(1, epochs + 1):
            epoch_loss = []
            start_time = time.time()
            for batch in self.train_iterator:
                X, X_len = batch.discourse_text
                X.to('cuda:0' if torch.cuda.is_available() else 'cpu')
                Y = batch.discourse_effectiveness.to(torch.int64).to('cuda:0' if torch.cuda.is_available() else 'cpu')
                Y_preds = self.model(X)
                loss = self.loss_fn(Y_preds, Y)
                epoch_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("Epoch {}/{}".format(i, epochs))
            print("Train Loss : {:.3f}".format(sum(epoch_loss) / len(epoch_loss)))
            losses.append(sum(epoch_loss) / len(epoch_loss))
            self.evaluate_loss_acc()
            end_time = time.time()
            print('Use time: {}\n'.format(secondsToStr(end_time - start_time)))

        return losses

    def evaluate_loss_acc(self, dataset="Valid", visualize=False):

        if dataset == "Valid":
            iterator = self.valid_iterator
        elif dataset == "Train":
            iterator = self.train_iterator
        else:
            assert False, "Wrong dataset with parameter dataset {}".format(dataset)

        if visualize:
            # Keep track of correct guesses in a confusion matrix
            confusion = torch.zeros(len(self.target_classes.vocab), len(self.target_classes.vocab))

        with torch.no_grad():
            Y_shuffled, Y_preds, losses = [], [], []
            for batch in iterator:
                X, X_len = batch.discourse_text
                X.to('cuda:0' if torch.cuda.is_available() else 'cpu')
                Y = batch.discourse_effectiveness.to(torch.int64).to('cuda:0' if torch.cuda.is_available() else 'cpu')
                Y_pred = self.model(X)
                loss = self.loss_fn(Y_pred, Y)
                losses.append(loss.item())

                Y_shuffled.append(Y)
                Y_preds.append(Y_pred.argmax(dim=-1))
                if visualize:
                    for y_id, y_pred_id in zip(Y, torch.argmax(Y_pred, axis=-1)):
                        confusion[y_id.item()][y_pred_id.item()] += 1

            Y_shuffled = torch.cat(Y_shuffled)
            Y_preds = torch.cat(Y_preds)
            print("{} Loss : {:.3f}".format(dataset, torch.tensor(losses).mean()))
            print("{} Acc : {:.3f}".format(dataset, accuracy_score(Y_shuffled.cpu().numpy(), Y_preds.cpu().numpy())))

        if visualize:
            # Normalize by dividing every row by its sum
            for i in range(len(self.target_classes.vocab)):
                confusion[i] = confusion[i] / confusion[i].sum()

            # Set up plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(confusion.numpy())
            fig.colorbar(cax)

            # Set up axes
            ax.set_xticklabels([''] + list(self.target_classes.vocab.stoi.keys()), rotation=90)
            ax.set_yticklabels([''] + list(self.target_classes.vocab.stoi.keys()))

            # Force label at every tick
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

            # sphinx_gallery_thumbnail_number = 2
            plt.show()


if __name__ == "__main__":
    fine_rnn = Fine_tune_rnn()
    fine_rnn.train()
    fine_rnn.evaluate_loss_acc(visualize=True)


