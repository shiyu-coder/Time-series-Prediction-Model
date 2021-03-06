from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_TSNE_STOCK, Dataset_BTCd, Dataset_BTC_1min, Dataset_BTC_15min
from exp.exp_basic import Exp_Basic
from models.model import DenseFormer
from tensorboardX import SummaryWriter
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')
writer = SummaryWriter(comment='transformer1_comment', filename_suffix="transformer1_suffix")
vali_test_count = 9
device_num = 6
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9'


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'denseformer': DenseFormer,
        }
        if self.args.model == 'denseformer':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                64,
            )

        model = model.double()
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
            model = model.cuda()
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'custom': Dataset_Custom,
            'TSNE': Dataset_TSNE_STOCK,
            'TSNE2': Dataset_TSNE_STOCK,
            'TSNE3': Dataset_TSNE_STOCK,
            'TSNE4': Dataset_TSNE_STOCK,
            'TSNE5': Dataset_TSNE_STOCK,
            'TSNE6': Dataset_TSNE_STOCK,
            'BTCd': Dataset_BTCd,
            'BTCd2': Dataset_BTCd,
            'BTCm1': Dataset_BTC_1min,
            'BTCm1f': Dataset_BTC_1min,
            'BTCm15i': Dataset_BTC_15min,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size * device_num,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            global vali_test_count
            vali_test_count += 1

            batch_x = batch_x.double()
            batch_y = batch_y.double()

            batch_x_mark = batch_x_mark.double()
            batch_y_mark = batch_y_mark.double()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double()

            batch_x, batch_x_mark = batch_x.cuda(), batch_x_mark.cuda()
            dec_inp, batch_y_mark = dec_inp.cuda(), batch_y_mark.cuda()
            batch_y = batch_y.cuda()
            # encoder - decoder
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            # for inc in np.arange(24):
            #     # print(pred[0][inc].shape)
            #     writer.add_scalars("Prediction_" + str(int(vali_test_count / 10)), {"prediction": pred[0][inc][3]}, inc)
            #     writer.add_scalars("Prediction_" + str(int(vali_test_count / 10)), {"true": true[0][inc][3]}, inc)
            # if vali_test_count % 80 == 0:
            #     for inc in np.arange(24):
            # print(pred[0][inc].shape)
            # if len(pred[0].shape) > 1:
            #     writer.add_scalars("Prediction_" + str(int(vali_test_count / 10)),
            #                        {"prediction": pred[0][inc][3]}, inc)
            #     writer.add_scalars("Prediction_" + str(int(vali_test_count / 10)), {"true": true[0][inc][3]},
            #                        inc)
            # else:
            #     writer.add_scalars("Prediction_" + str(int(vali_test_count / 10)),
            #                        {"prediction": pred[0][inc]}, inc)
            #     writer.add_scalars("Prediction_" + str(int(vali_test_count / 10)), {"true": true[0][inc]},
            #                        inc)
            # writer.add_scalars("Train loss", {"Train": loss}, total_count)
            # loss = criterion(pred[:, :, 3], true[:, :, 3])
            loss = criterion(pred, true)

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = './checkpoints/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        total_count = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                total_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.double()
                batch_y = batch_y.double()

                batch_x_mark = batch_x_mark.double()
                batch_y_mark = batch_y_mark.double()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double()

                batch_x, batch_x_mark = batch_x.cuda(), batch_x_mark.cuda()
                dec_inp, batch_y_mark = dec_inp.cuda(), batch_y_mark.cuda()
                batch_y = batch_y.cuda()
                # encoder - decoder
                # print(batch_x.shape)
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                # print(loss)

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                writer.add_scalars("Train loss", {"Train": loss}, total_count)
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            writer.add_scalars("Valid loss", {"Valid": vali_loss}, epoch + 1)
            writer.add_scalars("Test loss", {"Test": test_loss}, epoch + 1)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.double()
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double()
            batch_y_mark = batch_y_mark.double()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double()
            batch_x, batch_x_mark = batch_x.cuda(), batch_x_mark.cuda()
            dec_inp, batch_y_mark = dec_inp.cuda(), batch_y_mark.cuda()
            batch_y = batch_y.cuda()
            # encoder - decoder
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
