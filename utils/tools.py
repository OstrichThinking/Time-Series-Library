import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# def create_segment_list(ts_list, obs_win_len, pred_win_len, step_len=0):
#     """
#     Split a time series into segments of specified length.
    
#     Args:
#         ts_list: The time series data (list or array)
#         total_win_len: The length of each segment
        
#     Returns:
#         List of segments, each of length total_win_len
#     """

#     total_win_len = obs_win_len + pred_win_len

#     # Convert to numpy array for efficient slicing
#     ts_array = np.array(ts_list)
    
#     # Calculate number of complete segments
#     total_length = len(ts_array)
#     num_segments = total_length // total_win_len
    
#     segments = []
    
#     # Create segments
#     for i in range(num_segments):
#         start_idx = i * total_win_len
#         end_idx = start_idx + total_win_len
#         segment = ts_array[start_idx:end_idx]
#         segments.append(segment)

#     return segments

def create_segment_list(ts_list, obs_win_len, pred_win_len, step_len=0):
    """
    Split a time series into segments using a sliding window with specified step length.
    
    Args:
        ts_list: The time series data (list or array)
        obs_win_len: Length of the observation window
        pred_win_len: Length of the prediction window
        step_len: Step size for the sliding window (default=0 means equal to total_win_len)
        
    Returns:
        List of segments, each of length obs_win_len + pred_win_len
    """
    total_win_len = obs_win_len + pred_win_len

    # Convert to numpy array for efficient slicing
    ts_array = np.array(ts_list)
    
    # Calculate total length of the time series
    total_length = len(ts_array)
    
    # If step_len is 0 or invalid, set it to total_win_len (non-overlapping segments)
    if step_len <= 0:
        step_len = total_win_len
    
    # Calculate number of segments based on sliding window
    if total_length < total_win_len:
        return []  # Return empty list if time series is shorter than window
    num_segments = (total_length - total_win_len) // step_len + 1
    
    segments = []
    
    # Create segments using sliding window
    for i in range(num_segments):
        start_idx = i * step_len
        end_idx = start_idx + total_win_len
        # Ensure we don't exceed the array bounds
        if end_idx <= total_length:
            segment = ts_array[start_idx:end_idx]
            segments.append(segment)
    
    return segments
    