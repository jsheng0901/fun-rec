import torch
import torch.nn as nn
import numpy as np


class FeaturesEmbedding(nn.Module):
    """
    Embedding each features each value to embed dimensions
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # field_dims: list record each feature number of unique value
        # ex: field_dims = [2, 3, 4, 5], embedding lookup vocab size will be sum(field_dims) -> 14, like [14, 10]
        # which means transfer each features each unique value into embed dimes
        # same as one-hot each feature and then apply embedding, but here all unique feature will be used in same vocab
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # ex: field_dims = [2, 3, 4, 5], cum_sum -> [2, 5, 9, 14], [:-1]-> [2, 5, 9] -> add 0 then offsets [0, 2, 5, 9]
        # first feature not need add shift since first no shift from left side
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        # init parameter data
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, num_fields, embedding_dim]
        """
        # x -> [batch_size, num_fields] + [num_fields] -> [batch_size, num_fields]
        x = x + x.new_tensor(self.offsets)
        #  x * embedding -> [batch_size, num_fields] ->
        #  [batch_size, num_fields, sum(num_fields)] * [sum(num_fields), embed_dim]
        #  -> [batch_size, num_fields, embedding_dim]
        # same as nlp embedding layer, x input num_fields is equal length in nlp, and each is one is index in vocab,
        # sum(num_fields) is vocab size, then we need list each feature in vocab where is 1, and then do look up.
        return self.embedding(x)


class FeaturesLinear(torch.nn.Module):
    """
    Feature linear layer, same as fully connected linear layer. output = w_i * x + w_0
    """
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        # same as simple linear layer but output dim is 1 -> [sum(num_fields), 1], ex: [241895, 1],
        self.linear = torch.nn.Embedding(sum(field_dims), output_dim)
        # bias -> [1, ], will be broadcasting on each feature
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # same as features embedding layer, ex: [39, ]
        # use to shift each feature index value by previous all unique value size, transfer index to vocab index
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields] -> [4096, 39], each feature will be an index like in nlp token index in vocab.
        # but in here each feature index not cumulate by feature, ex: feature1: [0, 1, 2], features2: [0, 1]
        # but before pass into embedding, each unique value should be transfer to index in all vocab size, so
        # after shift each by each feature sum of unique value, we will have ex: feature1: [0, 1, 2], features2: [3, 4]
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # x * linear -> [batch_size, num_fields] -> [batch_size, num_fields, sum(num_fields)] * [sum(num_fields), 1]
        # -> [batch_size, num_fields, 1] -> sum -> [batch_size, 1] + bias -> [batch_size, 1] + [1, ] -> [batch_size, 1]
        # here sum across all features in each batch, same as linear layer, for each feature times weight then sum
        return torch.sum(self.linear(x), dim=1) + self.bias
