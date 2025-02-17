import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from collections import defaultdict

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            # 去掉data列
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # 只有一个csv，因此border1s[0]:border2s[0]就是训练集，每次使用训练集fit scaler
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # 使用训练集拟合好的scaler对其他数据集进行标准化
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # 时间戳
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class VitalDBLoader(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, fitted_scaler=None, timeenc=0, freq='h',
                 seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # 训练:验证:测试 比例为 7:1:2 
        assert flag in ['train', 'val', 'test',]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.static_features = args.static_features
        self.dynamic_features = args.dynamic_features
        
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # Initialize scalers for each feature to be standardized
        self.scalers = {feature: StandardScaler() for feature in self.static_features if feature != 'caseid' and feature != 'sex'}
        self.scalers.update({feature: StandardScaler() for feature in self.dynamic_features})
        
        self.fitted_scaler = fitted_scaler

        self.__read_data__()

    def __read_data__(self):
        # 只加载指定的列
        columns_to_read = self.static_features + self.dynamic_features
        df_raw = pd.read_csv(
            os.path.join(self.root_path, str(self.data_path)), 
            usecols=columns_to_read, nrows=5000)    # 调试时添加，nrows=5000

        # 按照caseid进行拆分，确保同一caseid的样本不会出现在不同的数据集中
        unique_caseids = df_raw['caseid'].unique()
        n_caseids = len(unique_caseids)
        train_cut = int(n_caseids * 0.7)
        val_cut = train_cut + int(n_caseids * 0.1)
        if self.set_type == 0:
            selected_caseids = unique_caseids[:train_cut]
        elif self.set_type == 1:
            selected_caseids = unique_caseids[train_cut:val_cut]
        elif self.set_type == 2:
            selected_caseids = unique_caseids[val_cut:]
        df_raw = df_raw[df_raw['caseid'].isin(selected_caseids)]

        self.__process_data(df_raw)

    def __process_data(self, data):

        def parse_sequence(sequence_str):
            sequence_str = sequence_str[1:-1]
            sequence_array = sequence_str.split(', ')

            # 均值填充 nan
            sequence_array = [np.nan if x == 'nan' else float(x) for x in sequence_array]
            mean_value = round(np.nanmean(sequence_array), 2)
            
            sequence_array_filled = np.where(np.isnan(sequence_array), mean_value, sequence_array)
            return sequence_array_filled
        
        examples = defaultdict(list)
        for index, row in data.iterrows():
            for feature in self.static_features:
                if feature != 'caseid':
                    examples[feature].append(row[feature])
            
            for feature in self.dynamic_features:
                if feature == 'prediction_maap':
                    # 只取前self.pred_len的数据
                    sequence_list = np.array(parse_sequence(row[feature]))
                    sequence_list = sequence_list[:self.pred_len]
                    sequence_str = ', '.join(map(str, sequence_list))
                    examples[feature].append(np.array(parse_sequence(sequence_str)))
                else:
                    examples[feature].append(np.array(parse_sequence(row[feature])))

        if self.scale and self.set_type == 0:
            print("Fitting scalers on training data...")
            for feature in self.static_features:
                if feature != 'caseid' and feature != 'sex':
                    self.scalers[feature].fit(np.array(examples[feature]).reshape(-1, 1))
            # 初始使用训练集拟合标准化 scaler
            for feature in self.dynamic_features:
                if feature in self.scalers:
                    self.scalers[feature].fit(examples[feature])
        else:
            # 测试和验证时，使用拟合好的 scaler
            self.scalers = self.fitted_scaler

        if self.scale:
            print("Transforming data with fitted scalers...")
            for feature in self.static_features:
                if feature != 'caseid' and feature != 'sex':
                    examples[feature] = self.scalers[feature].transform(np.array(examples[feature]).reshape(-1, 1))
            for feature in self.dynamic_features:
                if feature in self.scalers:
                    examples[feature] = self.scalers[feature].transform(examples[feature])
        
        self.data = examples

    def __getitem__(self, index):
        if self.features == 'S': # 单变量时序预测
            if 'Solar8000/ART_MBP_window_sample' in self.scalers:
                mbp = self.data['Solar8000/ART_MBP_window_sample'][index]
            if 'Solar8000/NIBP_MBP_window_sample' in self.scalers:
                mbp = self.data['Solar8000/NIBP_MBP_window_sample'][index]
            seq_x = np.stack([mbp], axis=1)

        else: # 'MS' 'M' 多变量时序预测
            seq_x = []
            for feature in self.static_features:
                if feature != 'caseid':
                    seq_x.append(np.full(self.seq_len, self.data[feature][index]))
            
            for feature in self.dynamic_features:
                if feature == 'prediction_maap':
                    continue
                seq_x.append(self.data[feature][index])

            seq_x = np.stack(seq_x, axis=1)

        # 预测的目标数据是 prediction_maap 和当前的 mbp，构建 seq_y
        prediction_maap = self.data['prediction_maap'][index]
        seq_y = prediction_maap[:, np.newaxis]  

        # 随机生成 seq_x_mark 和 seq_y_mark
        seq_x_mark = np.random.rand(*seq_x.shape)
        seq_y_mark = np.random.rand(*seq_y.shape)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data[self.dynamic_features[0]])

    def inverse_transform(self, data, flag='y'):
        if flag == 'y':
            return self.scalers['prediction_maap'].inverse_transform(data)
        else:
            # 检查两个可能的键，返回一个包含两个结果的字典
            results = {}
            if 'Solar8000/ART_MBP_window_sample' in self.scalers:
                results['ART_MBP'] = self.scalers['Solar8000/ART_MBP_window_sample'].inverse_transform(data)
            if 'Solar8000/NIBP_MBP_window_sample' in self.scalers:
                results['NIBP_MBP'] = self.scalers['Solar8000/NIBP_MBP_window_sample'].inverse_transform(data)
            
            if results:
                return results
            else:
                raise ValueError("Neither 'Solar8000/ART_MBP_window_sample' nor 'Solar8000/NIBP_MBP_window_sample' found in scalers.")
    
    def _get_train_scaler(self):
        return self.scalers


class VitalDBLoader_backup(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, fitted_scaler=None, timeenc=0, freq='h',
                 seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # 训练:验证:测试 比例为 7:1:2 
        assert flag in ['train', 'val', 'test',]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.static_features = args.static_features
        self.dynamic_features = args.dynamic_features
        
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # Initialize scalers for each feature to be standardized
        self.scaler_dbp = StandardScaler()
        self.scaler_sbp = StandardScaler()
        self.scaler_mbp = StandardScaler()
        self.scaler_bt = StandardScaler()
        self.scaler_hr = StandardScaler()

        # 用药
        self.scaler_ppf20_ce = StandardScaler()
        self.scaler_ppf20_cp = StandardScaler()
        self.scaler_ppf20_ct = StandardScaler()
        self.scaler_ppf20_rate = StandardScaler()
        # self.scaler_ppf20_vol = StandardScaler()
        self.scaler_rftn20_ce = StandardScaler()
        self.scaler_rftn20_cp = StandardScaler()
        self.scaler_rftn20_ct = StandardScaler()
        self.scaler_rftn20_rate = StandardScaler()
        # self.scaler_rftn20_vol = StandardScaler()

        # 呼吸相关
        self.scaler_etco2 = StandardScaler()
        self.scaler_feo2 = StandardScaler()
        self.scaler_fio2 = StandardScaler()
        self.scaler_inco2 = StandardScaler()
        self.scaler_vent_mawp = StandardScaler()
        self.scaler_vent_mv = StandardScaler()
        self.scaler_vent_rr = StandardScaler()

        self.scaler_prediction_maap = StandardScaler()
        
        self.fitted_scaler = fitted_scaler

        self.__read_data__()

    def __read_data__(self):

        # 定义需要读取的列列表
        columns_to_read = [
            'caseid',
            'sex',
            'age',
            'bmi',
            'Solar8000/ART_DBP_window_sample',
            'Solar8000/ART_MBP_window_sample',
            'Solar8000/ART_SBP_window_sample',
            'Solar8000/BT_window_sample',
            'Solar8000/HR_window_sample',
            # 用药
            'Orchestra/PPF20_CE_window_sample',
            'Orchestra/PPF20_CP_window_sample',
            'Orchestra/PPF20_CT_window_sample',
            'Orchestra/PPF20_RATE_window_sample',
            # 'Orchestra/PPF20_VOL',
            'Orchestra/RFTN20_CE_window_sample',
            'Orchestra/RFTN20_CP_window_sample',
            'Orchestra/RFTN20_CT_window_sample',
            'Orchestra/RFTN20_RATE_window_sample',
            # 'Orchestra/RFTN20_VOL',
            # 呼吸相关
            'Solar8000/ETCO2_window_sample',
            'Solar8000/FEO2_window_sample',
            'Solar8000/FIO2_window_sample',
            'Solar8000/INCO2_window_sample',
            'Solar8000/VENT_MAWP_window_sample',
            'Solar8000/VENT_MV_window_sample',
            'Solar8000/VENT_RR_window_sample',
            # label
            'prediction_maap'
        ]

        # 只加载指定的列
        df_raw = pd.read_csv(
            os.path.join(self.root_path, str(self.data_path)), 
            usecols=columns_to_read)

        # 按照caseid进行拆分，确保同一caseid的样本不会出现在不同的数据集中
        unique_caseids = df_raw['caseid'].unique()
        n_caseids = len(unique_caseids)
        train_cut = int(n_caseids * 0.7)
        val_cut = train_cut + int(n_caseids * 0.1)
        if self.set_type == 0:
            selected_caseids = unique_caseids[:train_cut]
        elif self.set_type == 1:
            selected_caseids = unique_caseids[train_cut:val_cut]
        elif self.set_type == 2:
            selected_caseids = unique_caseids[val_cut:]
        df_raw = df_raw[df_raw['caseid'].isin(selected_caseids)]

        self.__process_data(df_raw)

    def __process_data(self, data):

        def parse_sequence(sequence_str):
            sequence_str = sequence_str[1:-1]
            sequence_array = sequence_str.split(', ')

            # 均值填充 nan
            sequence_array = [np.nan if x == 'nan' else float(x) for x in sequence_array]
            mean_value = round(np.nanmean(sequence_array), 2)
            
            sequence_array_filled = np.where(np.isnan(sequence_array), mean_value, sequence_array)
            return sequence_array_filled
        
        examples = defaultdict(list)
        for index, row in data.iterrows():
            # 提取静态特征
            for key in ['sex', 'age', 'bmi']:
                examples[key].append(row[key])

            # 提取用药数据
            med_columns = {
                'ppf20_ce': 'Orchestra/PPF20_CE_window_sample',
                'ppf20_cp': 'Orchestra/PPF20_CP_window_sample',
                'ppf20_ct': 'Orchestra/PPF20_CT_window_sample',
                'ppf20_rate': 'Orchestra/PPF20_RATE_window_sample',
                'rftn20_ce': 'Orchestra/RFTN20_CE_window_sample',
                'rftn20_cp': 'Orchestra/RFTN20_CP_window_sample',
                'rftn20_ct': 'Orchestra/RFTN20_CT_window_sample',
                'rftn20_rate': 'Orchestra/RFTN20_RATE_window_sample'
            }
            for key, col in med_columns.items():
                examples[key].append(np.array(parse_sequence(row[col])))

            # 提取呼吸相关数据
            resp_columns = {
                'etco2': 'Solar8000/ETCO2_window_sample',
                'feo2': 'Solar8000/FEO2_window_sample',
                'fio2': 'Solar8000/FIO2_window_sample',
                'inco2': 'Solar8000/INCO2_window_sample',
                'vent_mawp': 'Solar8000/VENT_MAWP_window_sample',
                'vent_mv': 'Solar8000/VENT_MV_window_sample',
                'vent_rr': 'Solar8000/VENT_RR_window_sample'
            }
            for key, col in resp_columns.items():
                examples[key].append(np.array(parse_sequence(row[col])))

            # 提取时序特征（基本生命体征和预测值）
            vital_columns = {
                'dbp': 'Solar8000/ART_DBP_window_sample',
                'mbp': 'Solar8000/ART_MBP_window_sample',
                'sbp': 'Solar8000/ART_SBP_window_sample',
                'bt':  'Solar8000/BT_window_sample',
                'hr':  'Solar8000/HR_window_sample',
                'prediction_maap': 'prediction_maap'
            }
            for key, col in vital_columns.items():
                examples[key].append(np.array(parse_sequence(row[col])))
                
        if self.scale and self.set_type == 0:
            print("Fitting scalers on training data...")
            # 初始使用训练集拟合标准化 scaler
            self.scaler_dbp.fit(examples['dbp'])
            self.scaler_sbp.fit(examples['sbp'])
            self.scaler_mbp.fit(examples['mbp'])
            self.scaler_bt.fit(examples['bt'])
            self.scaler_hr.fit(examples['hr'])
            # 用药
            self.scaler_ppf20_ce.fit(examples['ppf20_ce'])
            self.scaler_ppf20_cp.fit(examples['ppf20_cp'])
            self.scaler_ppf20_ct.fit(examples['ppf20_ct'])
            self.scaler_ppf20_rate.fit(examples['ppf20_rate'])
            self.scaler_rftn20_ce.fit(examples['rftn20_ce'])
            self.scaler_rftn20_cp.fit(examples['rftn20_cp'])
            self.scaler_rftn20_ct.fit(examples['rftn20_ct'])
            self.scaler_rftn20_rate.fit(examples['rftn20_rate'])
            # 呼吸相关
            self.scaler_etco2.fit(examples['etco2'])
            self.scaler_feo2.fit(examples['feo2'])
            self.scaler_fio2.fit(examples['fio2'])
            self.scaler_inco2.fit(examples['inco2'])
            self.scaler_vent_mawp.fit(examples['vent_mawp'])
            self.scaler_vent_mv.fit(examples['vent_mv'])
            self.scaler_vent_rr.fit(examples['vent_rr'])

            self.scaler_prediction_maap.fit(examples['prediction_maap'])
        else :
            # 测试和验证时，使用拟合好的 scaler
            self.scaler_dbp = self.fitted_scaler['dbp']
            self.scaler_sbp = self.fitted_scaler['sbp']
            self.scaler_mbp = self.fitted_scaler['mbp']
            self.scaler_bt = self.fitted_scaler['bt']
            self.scaler_hr = self.fitted_scaler['hr']
            # 用药
            self.scaler_ppf20_ce = self.fitted_scaler['ppf20_ce']
            self.scaler_ppf20_cp = self.fitted_scaler['ppf20_cp']
            self.scaler_ppf20_ct = self.fitted_scaler['ppf20_ct']
            self.scaler_ppf20_rate = self.fitted_scaler['ppf20_rate']
            self.scaler_rftn20_ce = self.fitted_scaler['rftn20_ce']
            self.scaler_rftn20_cp = self.fitted_scaler['rftn20_cp']
            self.scaler_rftn20_ct = self.fitted_scaler['rftn20_ct']
            self.scaler_rftn20_rate = self.fitted_scaler['rftn20_rate']
            # 呼吸相关
            self.scaler_etco2 = self.fitted_scaler['etco2']
            self.scaler_feo2 = self.fitted_scaler['feo2']
            self.scaler_fio2 = self.fitted_scaler['fio2']
            self.scaler_inco2 = self.fitted_scaler['inco2']
            self.scaler_vent_mawp = self.fitted_scaler['vent_mawp']
            self.scaler_vent_mv = self.fitted_scaler['vent_mv']
            self.scaler_vent_rr = self.fitted_scaler['vent_rr']

            self.scaler_prediction_maap = self.fitted_scaler['prediction_maap']

        if self.scale:
            print("Transforming data with fitted scalers...")

            examples['dbp'] = self.scaler_dbp.transform(examples['dbp'])
            examples['sbp'] = self.scaler_sbp.transform(examples['sbp'])
            examples['mbp'] = self.scaler_mbp.transform(examples['mbp'])
            examples['bt'] = self.scaler_bt.transform(examples['bt'])
            examples['hr'] = self.scaler_hr.transform(examples['hr'])

            examples['ppf20_ce'] = self.scaler_ppf20_ce.transform(examples['ppf20_ce'])
            examples['ppf20_cp'] = self.scaler_ppf20_cp.transform(examples['ppf20_cp'])
            examples['ppf20_ct'] = self.scaler_ppf20_ct.transform(examples['ppf20_ct'])
            examples['ppf20_rate'] = self.scaler_ppf20_rate.transform(examples['ppf20_rate'])
            examples['rftn20_ce'] = self.scaler_rftn20_ce.transform(examples['rftn20_ce'])
            examples['rftn20_cp'] = self.scaler_rftn20_cp.transform(examples['rftn20_cp'])
            examples['rftn20_ct'] = self.scaler_rftn20_ct.transform(examples['rftn20_ct'])
            examples['rftn20_rate'] = self.scaler_rftn20_rate.transform(examples['rftn20_rate'])
            # 呼吸相关
            examples['etco2'] = self.scaler_etco2.transform(examples['etco2'])
            examples['feo2'] = self.scaler_feo2.transform(examples['feo2'])
            examples['fio2'] = self.scaler_fio2.transform(examples['fio2'])
            examples['inco2'] = self.scaler_inco2.transform(examples['inco2'])
            examples['vent_mawp'] = self.scaler_vent_mawp.transform(examples['vent_mawp'])
            examples['vent_mv'] = self.scaler_vent_mv.transform(examples['vent_mv'])
            examples['vent_rr'] = self.scaler_vent_rr.transform(examples['vent_rr'])    

            examples['prediction_maap'] = self.scaler_prediction_maap.transform(examples['prediction_maap'])
        
        self.data = examples

    def __getitem__(self, index):
        if self.features == 'S': # 单变量时序预测
            mbp = self.data['mbp'][index]
            seq_x = np.stack([mbp], axis=1)

        else: # 'MS' 'M' 多变量时序预测
            dbp = self.data['dbp'][index]
            sbp = self.data['sbp'][index]
            mbp = self.data['mbp'][index]
            bt = self.data['bt'][index]
            hr = self.data['hr'][index]

            # 用药
            ppf20_ce = self.data['ppf20_ce'][index]
            ppf20_cp = self.data['ppf20_cp'][index]
            ppf20_ct = self.data['ppf20_ct'][index]
            ppf20_rate = self.data['ppf20_rate'][index]
            rftn20_ce = self.data['rftn20_ce'][index]
            rftn20_cp = self.data['rftn20_cp'][index]
            rftn20_ct = self.data['rftn20_ct'][index]
            rftn20_rate = self.data['rftn20_rate'][index]

            # 呼吸相关
            etco2 = self.data['etco2'][index]
            feo2 = self.data['feo2'][index]
            fio2 = self.data['fio2'][index]
            inco2 = self.data['inco2'][index]
            vent_mawp = self.data['vent_mawp'][index]
            vent_mv = self.data['vent_mv'][index]
            vent_rr = self.data['vent_rr'][index]

            # 将静态特征扩展到与时间序列相同的长度（seq_len）
            sex = np.full(len(dbp), self.data['sex'][index])
            age = np.full(len(dbp), self.data['age'][index])
            bmi = np.full(len(dbp), self.data['bmi'][index])

            seq_x = np.stack([sex, age, bmi, 
                              dbp, sbp, bt, hr, 
                              ppf20_ce, ppf20_cp, ppf20_ct, ppf20_rate, 
                              rftn20_ce, rftn20_cp, rftn20_ct, rftn20_rate, 
                              etco2, feo2, fio2, inco2, vent_mawp, vent_mv, vent_rr,
                              mbp], axis=1)

        # 预测的目标数据是 prediction_mbp 和当前的 mbp，构建 seq_y
        prediction_maap = self.data['prediction_maap'][index]
        seq_y = prediction_maap[:, np.newaxis]

        # 随机生成 seq_x_mark 和 seq_y_mark
        seq_x_mark = np.random.rand(*seq_x.shape)
        seq_y_mark = np.random.rand(*seq_y.shape)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data['dbp'])

    def inverse_transform(self, data, flag='y'):
        if flag == 'y':
            return self.scaler_prediction_maap.inverse_transform(data)
        else:
            return self.scaler_mbp.inverse_transform(data)
    
    def _get_train_scaler(self):
        return {
            'dbp': self.scaler_dbp,
            'sbp': self.scaler_sbp,
            'mbp': self.scaler_mbp,
            'bt': self.scaler_bt,
            'hr': self.scaler_hr,
            'ppf20_ce': self.scaler_ppf20_ce,
            'ppf20_cp': self.scaler_ppf20_cp,
            'ppf20_ct': self.scaler_ppf20_ct,
            'ppf20_rate': self.scaler_ppf20_rate,
            'rftn20_ce': self.scaler_rftn20_ce,
            'rftn20_cp': self.scaler_rftn20_cp,
            'rftn20_ct': self.scaler_rftn20_ct,
            'rftn20_rate': self.scaler_rftn20_rate,
            'etco2': self.scaler_etco2,
            'feo2': self.scaler_feo2,
            'fio2': self.scaler_fio2,
            'inco2': self.scaler_inco2,
            'vent_mawp': self.scaler_vent_mawp,
            'vent_mv': self.scaler_vent_mv,
            'vent_rr': self.scaler_vent_rr,

            'prediction_maap': self.scaler_prediction_maap
        }


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
