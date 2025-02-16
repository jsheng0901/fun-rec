import logging
import time
from sklearn import metrics
import torch
import torch.nn as nn
from tqdm import tqdm

from notes.ranking.ESMM.model import EntireSpaceMultitaskModel
from notes.utils.dataloader import create_dataset, BatchLoader
from config import EntireSpaceMultitaskModelConfig
from notes.utils.trainer import Trainer, EarlyStopper

logging.basicConfig(level=logging.INFO)


class ESMMTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, batch_size=None, num_tasks=1):
        super().__init__(model, optimizer, criterion, batch_size)
        self.num_tasks = num_tasks

    def train(self, train_x, train_y, epoch=100, trials=None, valid_x=None, valid_y=None):
        # if batch loader
        if self.batch_size:
            train_loader = BatchLoader(train_x, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_x, train_y]]

        if trials:
            self.early_stopper = EarlyStopper(self.model, trials, self.num_tasks)

        train_loss_list = []
        for step in tqdm(range(epoch)):
            t1 = time.time()
            # train part
            self.model.train()
            # accumulate loss by batch
            batch_train_loss = 0
            for b_x, b_y in train_loader:
                pred_y = self.model(b_x)
                # multitask loss simple sum together as one loss
                train_loss = [self.criterion(pred_y[:, i], b_y[:, i]) for i in range(self.num_tasks)]
                total_train_loss = sum(train_loss)
                self.optimizer.zero_grad()
                total_train_loss.backward()
                self.optimizer.step()
                # here loss already calculate avg in batch, so we need time batch size back to calculate total loss
                batch_train_loss += total_train_loss.detach() * len(b_x)

            # record each epoch avg loss
            train_loss_list.append(batch_train_loss / len(train_x))

            # valid mode, check early stopper or not
            if trials:
                # valid loss and metric
                valid_loss, valid_metric = self.test(valid_x, valid_y)
                # valid_loss_list.append(valid_loss)
                # train loss and metric
                train_loss, train_metric = self.test(train_x, train_y)
                t2 = time.time()
                dt = t2 - t1
                logging.info(f"step {step}, dt: {dt * 1000:.2f}ms, train total loss: {total_train_loss:.4f}")
                for i in range(self.num_tasks):
                    logging.info(
                        f"task{i + 1}: train loss: {train_loss[i]}, train metric: {train_metric[i]:.3f},"
                        f"val loss： {valid_loss[i]:.4f}, val auc: {valid_metric[i]:.3f}"
                    )
                if self.early_stopper.is_continuable(valid_metric) is False:
                    break

    def test(self, test_x, test_y):
        # eval mode
        self.model.eval()
        # calculate pred value and loss
        with torch.no_grad():
            pred_y = self.model(test_x)
            # for test record multitask loss separate
            test_loss = [self.criterion(pred_y[:, i], test_y[:, i]).detach() for i in range(self.num_tasks)]

        # calculate different task metric
        if self.task == 'classification':
            test_metric = [metrics.roc_auc_score(test_y[:, i].cpu(), pred_y[:, i].cpu()) for i in range(self.num_tasks)]
        if self.task == 'regression':
            test_metric = [-test_loss[i] for i in range(self.num_tasks)]

        return test_loss, test_metric


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
    num_tasks = config['num_tasks']

    # loading the data
    t1 = time.time()
    dataset = create_dataset('adult', sample_num=sample_size, device=device)
    field_dims, (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.train_valid_test_split()
    t2 = time.time()
    logging.info(f"Loading data takes {(t2 - t1) * 1000}ms")

    # init model
    torch.manual_seed(1337)
    model = EntireSpaceMultitaskModel(field_dims, embed_dim, mlp_dims, dropout).to(device)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    criterion = nn.BCELoss()

    # start train ESMM model
    trainer = ESMMTrainer(model, optimizer, criterion, batch_size, num_tasks)
    trainer.train(train_x, train_y, epoch=num_epochs, trials=trials, valid_x=valid_x, valid_y=valid_y)
    test_loss, test_auc = trainer.test(test_x, test_y)
    for i in range(trainer.num_tasks):
        logging.info(f"Entire Space Multi-task model task{i + 1}: "
                     f"test loss: {test_loss[i]:.5f} | test_auc: {test_auc[i]:.5f}")

    return


if __name__ == '__main__':
    logging.info('Start Entire Space Multi-task Model Train')
    dcn_config = EntireSpaceMultitaskModelConfig.all_config
    train(dcn_config)
