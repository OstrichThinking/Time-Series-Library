from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, ioh_classification_metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import random


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, fitted_scaler=None):
        data_set, data_loader, fitted_scaler = data_provider(
            self.args, 
            flag, 
            fitted_scaler=fitted_scaler,
        )
        return data_set, data_loader, fitted_scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif self.args.loss == 'Huber':
            criterion = nn.HuberLoss(delta=0.5)
        elif self.args.loss == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss(beta=0.5)
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if not self.args.use_embed:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader, scaler = self._get_data(flag='train', fitted_scaler=None)
        self.fitted_scaler = scaler
        vali_data, vali_loader, _ = self._get_data(flag='val', fitted_scaler=scaler)
        test_data, test_loader, _ = self._get_data(flag='test', fitted_scaler=scaler)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        print(f"Start training time: {time.strftime('%Y年%m月%d日 %H:%M:%S', time.localtime(time_now))}")

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if not self.args.use_embed:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())


                if (i + 1) % 10 == 0:
                    self.swanlab.log({"iter_loss": loss.item()})

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.swanlab.log({"epoch_loss": train_loss, "epoch_vali_loss": vali_loss, "epoch_test_loss": test_loss})

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.swanlab.finish()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader, _ = self._get_data(flag='test', fitted_scaler=self.fitted_scaler)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if not self.args.use_embed:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # output [batch_size, pred_len, 1]
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    
                    # 原版
                    # outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    outputs = test_data.inverse_transform(outputs.reshape(shape[0], shape[1]), flag='y').reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0], shape[1]), flag='y').reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # 原版
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        # 原版 input 是 [batch_size, seq_len, 8]
                        # input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)

                        input_mbp = input[:, :, -1]
                        # import pdb; pdb.set_trace()
                        input_mbp_inverse_dict = test_data.inverse_transform(input_mbp, flag='x')
                        
                        # 处理 ART_MAP 和 NIBP_MAP
                        input_mbp_inverse = None
                        if 'ART_MBP' in input_mbp_inverse_dict and 'NIBP_MBP' in input_mbp_inverse_dict:
                            # TODO 待做，有创和无创的平均动脉压均需要返回时如何处理？
                            pass
                        elif 'ART_MBP' in input_mbp_inverse_dict:
                            input_mbp_inverse = input_mbp_inverse_dict['ART_MBP']
                        elif 'NIBP_MBP' in input_mbp_inverse_dict:
                            input_mbp_inverse = input_mbp_inverse_dict['NIBP_MBP']
                    
                    # 原版
                    gt = np.concatenate((input_mbp_inverse[0, :], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_mbp_inverse[0, :], pred[0, :, -1]), axis=0)

                    # gt: [seq_len+pred_len,]
                    # pd: [seq_len+pred_len,]
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("波形预测性能比较:")
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format("MSE", "MAE", "RMSE"))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format(np.around(mse, decimals=5), np.around(mae, 5), np.around(rmse, 5)))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")

        auc, accuracy, recall, precision, specificity, F1, TP, FP, FN, TN = ioh_classification_metric(preds, trues, exp_stime=self.args.exp_stime)
        print("分类性能比较:")
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format("AUC", "Accuracy", "Recall"))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format(round(auc, 5), round(accuracy, 5), round(recall, 5)))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format("Precision", "Specificity", "F1"))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format(round(precision, 5), round(specificity, 5), round(F1, 5)))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("混淆矩阵:")
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format("TP", "FN", "--"))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format(TP, FN, '--'))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format("FP", "TN", "--"))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        print("|{:^20}|{:^20}|{:^20}|".format(FP, TN, '--'))
        print("+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+")
        
        time_now = time.time()
        print(f"Test completion time: {time.strftime('%Y年%m月%d日 %H:%M:%S', time.localtime(time_now))}")
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, precision, recall, F1, accuracy, specificity, auc]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
