import logging
import time
import torch
import torch.nn as nn

from notes.ranking.PNN.model import ProductNeuralNetworkModel
from notes.utils.dataloader import create_dataset
from config import PNNModelConfig
from notes.utils.trainer import Trainer

logging.basicConfig(level=logging.INFO)


def train(config):
    # load all parameters
    device = config['device']
    embed_dim = config['embed_dim']
    learning_rate = config['learning_rate']
    regularization = config['regularization']
    num_epochs = config['num_epochs']
    trials = config['trials']
    sample_size = config['sample_size']
    batch_size = config['batch_size']
    mlp_dims = config['mlp_dims']
    dropout = config['dropout']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('criteo', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init ipnn model
    torch.manual_seed(1337)
    ipnn_model = ProductNeuralNetworkModel(field_dims, mlp_dims, embed_dim, dropout, 'inner').to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(ipnn_model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train ipnn the model
    trainer = Trainer(ipnn_model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Ipnn model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    # init opnn model
    opnn_model = ProductNeuralNetworkModel(field_dims, mlp_dims, embed_dim, dropout, 'outer').to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(opnn_model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train opnn the model
    trainer = Trainer(opnn_model, optimizer, criterion, batch_size)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    logging.info(f"Opnn model test_loss:  {test_loss:.5f} | test_auc: {test_auc:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Product Neural Network Model Train')
    pnn_config = PNNModelConfig.all_config
    train(pnn_config)
