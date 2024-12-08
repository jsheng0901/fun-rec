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


class FeaturesLinear(nn.Module):
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


class FeaturesCross(nn.Module):
    """
    Feature cross layer, output = 0.5 * sum_f((sum_i(vi_f * x_i))^2 - sum_i(vi_f^2 * x_i^2))
    Calculate each feature cross interaction weight.
    """
    def __init__(self, field_dims, embed_dim, reduce_sum=True):
        super().__init__()
        self.embedding_layer = FeaturesEmbedding(field_dims, embed_dim)
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1] or [batch_size, embed_dim]
        """
        # x -> [batch_size, num_fields]
        # embedding -> [batch_size, num_fields, embedding_dim] -> [4096, 39, 10]
        x = self.embedding_layer(x)
        # formula (vi_f * x_i)^2 -> x * cross -> ([n_samples, n_features] * [n_features, k])^2 -> [n_samples, k]^2
        # x -> [batch_size, num_fields, embedding_dim] -> sum -> [batch_size, k]^2 -> [4096, 10]
        square_of_sum = torch.pow(torch.sum(x, dim=1), 2)
        # (vi_f^2 * x_i^2) -> x^2 * cross^2 -> ([n_samples, n_features])^2 * ([n_features, k])^2 -> [n_samples, k]
        # x -> [batch_size, num_fields, embedding_dim]^2 -> sum -> [batch_size, k] -> [4096, 10]
        sum_of_square = torch.sum(torch.pow(x, 2), dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            # each sample sum along k dims, sum([batch_size, k] - [batch_size, k]) -> [batch_size, 1] -> [4096, 1]
            output = torch.sum(output, dim=1, keepdim=True)
        return 0.5 * output


class MultiLayerPerceptron(nn.Module):
    """
    Feed forward neural network layer, same name as mlp layer. l1 = w_1 * x + w_1_0, l2 = w_2 * l1 + w_2_0,
    """
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        # stack the layer
        layers = list()
        # loop through all layer output embedding dims
        for embed_dim in embed_dims:
            # add linear layer -> [input_dime, embed_dim]
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # add batch norm layer on embed dim
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            # add relu activation function
            layers.append(torch.nn.ReLU())
            # add drop out layer
            layers.append(torch.nn.Dropout(p=dropout))
            # last layer output dime is next layer input dime
            input_dim = embed_dim
        # if we have output final layer, then add one linear layer for output
        if output_layer:
            # add output linear layer -> [input_dime, 1]
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: [batch_size, embed_dim]
        :return: if have output layer [batch_size, 1], else [batch_size, embed_dim]
        """
        # x -> [batch_size, embed_dim]
        # [batch_size, embed_dim] * [embed_dim, embed_dim_1] * [embed_dim, embed_dim_2] ... -> [batch_size, embed_dim]
        return self.mlp(x)


class EmbeddingsInteraction(nn.Module):
    """
    Embedding interaction layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: [batch_size, num_fields, embedding_dim]
        :return: [batch_size, num_fields*(num_fields)//2, embedding_dim]
        """
        num_fields = x.shape[1]
        row, col = [], []
        # loop through all feature combination, each feature need calculate inner product with all other feature embed
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                row.append(i)
                col.append(j)
        # x1: [batch_size, 1, embedding_dim] * x2: [batch_size, 1, embedding_dim] -> [batch_size, 1, embedding_dim]
        # we have num_fields * (num_fields)//2 combination, mul will keep same output shape as input
        # then we have [batch_size, num_fields * (num_fields)//2, embedding_dim] output
        interaction = torch.mul(x[:, row], x[:, col])

        return interaction
