import torch
from torch import nn
import torch.nn.functional as F

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
        hidden_input = torch.zeros(batch_x.shape[0], self.hidden_size)

        # input(batch_x.shape)
        for word_id in range(batch_x.shape[-1]):
            embeddings = self.embedding_layer(batch_x[:, word_id])
            # print("embedding: ", str(embeddings.shape))
            # print("hidden: ", str(hidden_input.shape))
            combined_input = torch.cat((embeddings, hidden_input), 1)
            # print("combined: ", str(combined_input.shape))
            hidden_input = self.i2h(combined_input)
            # input(hidden_input.shape)
        output = self.i2o(combined_input)
        output = self.softmax(output)

        return output

    def get_loss_fn(self):
        # return nn.BCELoss()
        return nn.NLLLoss()

    def get_optimizer(self, params=None, lr=1e-3):
        assert params != None
        return torch.optim.SGD(params, lr=lr)



