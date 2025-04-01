import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.metrics import Check_If_IOH

def read_data(file_path, flag):

    # 从文件读取 JSONL 数据
    case_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 逐行读取并解析 JSON
                try:
                    case = json.loads(line.strip())
                    case_list.append(case)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
                    continue  # 跳过有错误的行
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return

    # 计算数据集分割点
    n_caseids = len(case_list)
    train_cut = int(n_caseids * 0.7)
    test_cut = train_cut + int(n_caseids * 0.2)

    # 根据 falg 分割数据集
    if flag == 'train':
        case_subset = case_list[:train_cut]
    elif flag == 'test':
        case_subset = case_list[train_cut:test_cut]
    else:
        case_subset = case_list[test_cut:]

    return case_subset


def convert_jsonl_to_sample_list(case_subset, flag,
                      seq_len, pred_len, s_win, stime, exp_stime, 
                      static_features, dynamic_features):
    
    sample_list = defaultdict(list)
    valid_samples_indices = []
    
    for case in tqdm(case_subset[:]):
        # 处理时序变量
        case_sample_num = 0
        for feature in dynamic_features:
            if feature != 'prediction_maap' and feature != 'seq_time_stamp_list' and feature != 'pred_time_stamp_list':
                # 对每个时序变量进行滑动窗口处理
                case_sample_list = create_segment_list(
                    ts_list=case[feature], 
                    obs_win_len=seq_len, 
                    pred_win_len=pred_len, 
                    step_len=s_win, 
                    stime=stime,
                    exp_stime=exp_stime
                )
                sample_list[feature].extend([item[:seq_len] for item in case_sample_list])
                # 如果为目标变量，将其加入到预测目标列表
                if feature == 'Solar8000/ART_MBP':
                    sample_list['prediction_maap'].extend([item[-pred_len:] for item in case_sample_list])
                
                # 记录病例样本数量
                case_sample_num = len(case_sample_list)
        
        # 处理静态变量
        case_timestamp_list = []
        for feature in static_features:
            if feature != 'caseid' and feature != 'time':
                # 按照case切分的样本数重复静态变量的数量
                case_static_list = np.full(case_sample_num, case[feature])
                # 将静态变量对齐时序时间步长 seq_len
                case_seq_static_list = [np.full(seq_len, item) for item in case_static_list]
                sample_list[feature].extend(case_seq_static_list)
            if feature == 'time':
                times = case[feature].split('-')
                start = int(times[0])
                end = int(times[1]) 
                case_timestamp_list = np.arange(start, end + stime, stime)
                
        # 添加 time stamp
        timestamp_list = create_segment_list(
            ts_list=case_timestamp_list,
            obs_win_len=seq_len,
            pred_win_len=pred_len,
            step_len=s_win,
            stime=stime,
            exp_stime=exp_stime
        )
        sample_list['seq_time_stamp_list'].extend([item[:seq_len] for item in timestamp_list])
        sample_list['pred_time_stamp_list'].extend([item[-pred_len:] for item in timestamp_list])
    
    # check ts vars 
    print(flag + ' before diff 50 :', len(sample_list[dynamic_features[0]]))
    sample_list = check_sample_valid(sample_list, dynamic_features)
    print(flag + ' after diff 50 :', len(sample_list[dynamic_features[0]]))
    
    # check ioh lable
    lebel_list = []
    for i in range(len(sample_list['prediction_maap'])):
        if Check_If_IOH(sample_list['prediction_maap'][i], 65, int(60/exp_stime)):
            lebel_list.append(1)
        else:
            lebel_list.append(0)
    
    sample_list['label'].extend(lebel_list)  # Add lebel_list to sample_list['label']
    return sample_list


def create_segment_list(ts_list, obs_win_len, pred_win_len, step_len, stime, exp_stime):
   
    # 数据集数据点采样点间隔与实验设置不同 ———> 需要降采样
    if stime != exp_stime:
        ts_list = ts_list[::exp_stime//stime]
    
    total_win_len = obs_win_len + pred_win_len
    ts_array = np.array(ts_list)
    total_length = len(ts_array)
    
    if total_length < total_win_len:
        return []
    
    num_segments = (total_length - total_win_len) // step_len + 1
    segments = []
    
    for i in range(num_segments):
        start_idx = i * step_len
        end_idx = start_idx + total_win_len
        if end_idx <= total_length:
            segment = ts_array[start_idx:end_idx]
            segments.append(segment)
    
    return segments
    
def check_sample_valid(sample_list, dynamic_features):
    
    valid_samples_indices = []
    # 检查每个样本是否有突变大于50的序列
    valid_sample_list = defaultdict(list)
    current_index = 0
    total_samples = len(sample_list[next(iter(sample_list))])
    
    for i in range(total_samples):
        valid_sample = True
        
        # 检查每个动态特征
        for feature in dynamic_features:
            if feature != 'prediction_maap' and feature != 'seq_time_stamp_list' and feature != 'pred_time_stamp_list':
                # 计算差分
                feature_diffs = np.abs(np.diff(sample_list[feature][i]))
                # 检查是否有突变大于50
                if np.any(feature_diffs > 50):
                    valid_sample = False
                    break
        
        # 如果样本有效，添加到有效样本列表
        if valid_sample:
            for key in sample_list:
                valid_sample_list[key].append(sample_list[key][i])
            valid_samples_indices.append(current_index)
        
        current_index += 1
    
    return valid_sample_list
    