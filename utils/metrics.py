import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def ioh_classification_metric(pred, true, IOH_value=65, stime=20):
    
    # 15分钟预5分钟，两秒一个点 sql_len:450 pred_len:150
    # map 低于65，持续1分钟，则认为发生IoH
    # IOH_value=65
    # stime 采样间隔
    
    pred_labels = []
    true_labels = []

    # 计算一分钟持续的点数
    duration = 60 / stime
    
    for i in range(len(pred)):
        pred_labels.append(Check_If_IOH(pred[i], IOH_value, duration))
        true_labels.append(Check_If_IOH(true[i], IOH_value, duration))
    
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    
    # 计算精确率、召回率、F1分数、准确率、特异性
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0
    
    # 计算AUC
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, pred_labels)
    else:
        auc = float('nan')  # 或者选择一个合适的默认值  #TODO
    
    return precision, recall, F1, accuracy, specificity, auc
    

def Check_If_IOH(time_series, IOH_value, duration):
    """
    Check if there is a period of intraoperative hypotension (IOH) in the time series.

    Parameters:
    - time_series (1D array-like or torch.Tensor): The blood pressure time series (NumPy array or PyTorch tensor).
    - IOH_value (float): Threshold value for IOH (blood pressure below this is considered hypotensive).
    - duration (float): Duration in seconds or samples that defines IOH.

    Returns:
    - bool: True if IOH is detected, otherwise False.
    """
    # 将 duration 转换为采样点数（假设 duration 已为采样点数，若需采样率，需额外参数）
    duration_samples = int(duration)

    # 判断输入类型并转换为适当格式
    if isinstance(time_series, np.ndarray):
        # CPU 上使用 NumPy
        time_series = time_series  # 已经是 NumPy 数组
        is_torch = False
    elif isinstance(time_series, torch.Tensor):
        # GPU 或 CPU 上使用 PyTorch
        time_series = time_series  # 已经是 PyTorch 张量
        is_torch = True
    else:
        raise TypeError("time_series must be a NumPy array or PyTorch tensor")

    # 如果时间序列长度小于 duration_samples，不可能满足 IOH 条件
    if len(time_series) < duration_samples:
        return False

    # 创建低于阈值的布尔掩码
    if is_torch:
        below_threshold = time_series < IOH_value  # PyTorch 张量比较
        # 使用卷积检查连续 duration_samples 个 True
        kernel = torch.ones(duration_samples, device=time_series.device)
        conv_result = torch.conv1d(below_threshold.float().squeeze(-1).unsqueeze(0).unsqueeze(0), 
                                  kernel.unsqueeze(0).unsqueeze(0), 
                                  padding=0)
        # 如果卷积结果等于 duration_samples，说明有连续 duration_samples 个 True
        return torch.any(conv_result == duration_samples).item()
    else:
        below_threshold = time_series < IOH_value  # NumPy 数组比较
        # 使用滑动窗口检查
        for i in range(len(below_threshold) - duration_samples + 1):
            if np.all(below_threshold[i:i + duration_samples]):
                return True
        return False
    
def Check_If_IOH_permin(time_series, IOH_value):
    return time_series < IOH_value