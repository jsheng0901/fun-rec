import logging
import time

import torch
from sklearn import metrics
from tqdm import tqdm

from notes.utils.dataloader import BatchLoader


class Trainer:

    def __init__(self, model, optimizer, criterion, batch_size=None, task='classification'):
        assert task in ['classification', 'regression']
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.task = task

    def train(self, train_x, train_y, epoch=100, eval_interval=10, valid_x=None, valid_y=None):
        # if batch loader
        if self.batch_size:
            train_loader = BatchLoader(train_x, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_x, train_y]]

        # if trials:
        #     early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []

        for step in tqdm(range(epoch)):
            t1 = time.time()
            # train mode
            self.model.train()
            # accumulate loss by batch
            batch_train_loss = 0
            for b_x, b_y in train_loader:
                pred_y = self.model(b_x)
                train_loss = self.criterion(pred_y, b_y)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                # here loss already calculate avg in batch, so we need time batch size back to calculate total loss
                batch_train_loss += train_loss.detach() * len(b_x)

            # record each epoch avg loss
            train_loss_list.append(batch_train_loss / len(train_x))

            # valid mode
            if step % eval_interval == 0 or step == epoch - 1:
                # if trials:
                # valid loss and metric
                valid_loss, valid_metric = self.test(valid_x, valid_y)
                # valid_loss_list.append(valid_loss)
                # train loss and metric
                train_loss, train_metric = self.test(train_x, train_y)
                t2 = time.time()
                dt = t2 - t1
                logging.info(f"step {step}, dt: {dt * 1000:.2f}ms, train loss: {train_loss:.4f},"
                             f"train metric: {train_metric:.2f}, val loss： {valid_loss:.4f},"
                             f"val auc: {valid_metric:.2f}")
                # if not early_stopper.is_continuable(valid_metric):
                #     break

        # if trials:
        #     self.model.load_state_dict(early_stopper.best_state)
        #     plt.plot(valid_loss_list, label='valid_loss')
        #
        # plt.plot(train_loss_list, label='train_loss')
        # plt.legend()
        # plt.show()

        # print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_X, train_y)))

        # if trials:
        #     print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_X, valid_y)))

    def test(self, test_x, test_y):
        # eval mode
        self.model.eval()
        # calculate pred value and loss
        with torch.no_grad():
            pred_y = self.model(test_x)
            test_loss = self.criterion(pred_y, test_y).detach()

        # calculate different task metric
        if self.task == 'classification':
            test_metric = metrics.roc_auc_score(test_y.cpu(), pred_y.cpu())
        if self.task == 'regression':
            test_metric = -test_loss

        return test_loss, test_metric