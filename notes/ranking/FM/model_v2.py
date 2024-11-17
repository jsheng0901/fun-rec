import torch.nn as nn
from notes.ranking.layer.layer import FeaturesEmbedding, FeaturesLinear
import torch


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
        :return: [batch_size, 1]
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


class FactorizationMachine(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    This version linear and cross layer use embedding layer before pass into model.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear_layer = FeaturesLinear(field_dims)
        self.cross_layer = FeaturesCross(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_fields]
        :return: [batch_size, 1]
        """
        # x -> [batch_size, num_fields]
        # linear -> [batch_size, 1]
        linear = self.linear_layer(x)
        # cross -> [batch_size, 1]
        cross = self.cross_layer(x)
        # output [batch_size, 1]
        output = linear + cross

        # apply sigmoid to transfer to probability, no squeeze here since target is same [batch_size, 1] size
        return torch.sigmoid(output)
