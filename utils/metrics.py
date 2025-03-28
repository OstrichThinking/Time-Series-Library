import numpy as np
from sklearn.metrics import roc_auc_score

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

def ioh_classification_metric(pred, true, IOH_value=65, exp_stime=2):
    
    # 15分钟预5分钟，两秒一个点 sql_len:450 pred_len:150
    # map 低于65，持续1分钟，则认为发生IoH
    # IOH_value=65
    # stime 采样间隔
    
    pred_labels = []
    true_labels = []

    # 计算一分钟持续的点数
    duration = 60 / exp_stime
    
    for i in range(len(pred)):
        pred_labels.append(ground_truth_labeling(pred[i], len(pred[i])))
        true_labels.append(pred_labeling(true[i], duration, len(true[i])))
    
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

    return auc, accuracy, recall, precision, specificity, F1, TP, FP, FN, TN
    

def Check_If_IOH(time_series, IOH_value, duration):
    """
    Check if there is a period of intraoperative hypotension (IOH) in the time series.

    Parameters:
    - time_series (1D array-like): The blood pressure time series.
    - srate: Sampling rate of the time series (samples per second).
    - IOH_value: Threshold value for IOH (blood pressure below this is considered hypotensive).
    - duration: duration in seconds that defines IOH (must stay below IOH_value for this period).

    Returns:
    - bool: True if IOH is detected, otherwise False.
    """
    # 将Duration转换为采样点数
    duration_samples = int(duration)  # 将duration转换为整数
    
    # 如果时间序列长度小于duration_samples，不可能满足IOH条件，直接返回False
    if len(time_series) < duration_samples:
        return False
    
    # 创建一个布尔掩码数组，标记低于IOH阈值的点
    below_threshold = time_series < IOH_value
    
    # 使用滑动窗口检查是否存在连续的duration_samples个值都低于IOH_value
    for i in range(len(below_threshold) - duration_samples + 1):
        # 检查当前滑动窗口内的所有值是否都为True（即都低于IOH_value）
        if np.all(below_threshold[i:i + duration_samples]):
            return True
    
    return False


def ground_truth_labeling(MAP_actual, t, T_seq, theta_MAP=65):
    
    # from
    # HMF: A Hybrid Multi-Factor Framework for Dynamic Intraoperative Hypotension Prediction
    
    J_actual = 0
    L_actual = np.zeros(T_seq, dtype=int)

    for i in range(T_seq - t + 1):
        MAP_max = np.max(MAP_actual[i : i + t])
        if MAP_max <= theta_MAP:
            L_actual[i : i + t] = 1
        
        sum_actual = np.sum(L_actual[i : i + t])
        if sum_actual > 0:
            J_actual = 1

    return J_actual

def pred_labeling(MAP_pred, t, T_seq, theta_MAP=65):
    
    # from 
    # HMF: A Hybrid Multi-Factor Framework for Dynamic Intraoperative Hypotension Prediction
    
    J_prediction = 0

    for i in range(T_seq - t + 1):
        sum_prediction = np.sum(MAP_pred[i : i + t] <= theta_MAP)
        if sum_prediction > 0.8 * t:
            J_prediction = 1

    return  J_prediction
    
    
    
    
    