import os
import numpy as np
from collections import defaultdict
from data_augmentor import oversample_minority
from utils.metrics import Check_If_IOH

class IOHDataAugmentor(object):
    def __init__(self, args, X: defaultdict, method='smote'):
        self.args = args
        self.X = X
        self.y = None
        self.method = method
        self.augmentor_dict = {
            'oversample_minority': oversample_minority
        }

    def _get_ioh_label(self, IOH_value=65, stime=20):
        duration = 60 / stime
        prediction_maap = self.X['prediction_maap']
        y = []
        for i in range(len(prediction_maap)):   # len(prediction_maap) 为样本数量
            y.append(Check_If_IOH(prediction_maap[i], IOH_value, duration))
        self.y = np.array(y)
        return self.y

    # TODO 其他数据增强方法可能用到
    def _flatten_time_series(self, X):
        """将时间序列数据展平为二维"""
        n_samples = len(X)
        n_timesteps = len(X[0])  # 假设所有样本时间步长一致
        return np.array(X).reshape(n_samples, n_timesteps)

    def _reshape_to_time_series(self, X_flat, n_timesteps):
        """将展平的数据恢复为时间序列格式"""
        n_samples = X_flat.shape[0]
        return X_flat.reshape(n_samples, n_timesteps)

    # TODO 其他数据增强方法可能用到
    def _build_augmentor(self):
        augmentor = self.augmentor_dict[self.args.augment_method].Augmentor(self.args)
        return augmentor
    
    def augment(self, X, y):
        pass