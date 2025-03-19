import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 生成示例多变量时间序列数据
def generate_sample_data(n_samples=100, n_timesteps=10, n_features=3):
    # 创建一个三维数组 [样本数, 时间步长, 特征数]
    data = np.random.randn(n_samples, n_timesteps, n_features)
    # 创建二分类标签
    labels = np.random.randint(0, 2, size=(n_samples,))
    return data, labels

# 将多变量时间序列数据展平为2D以适应SMOTE
def flatten_time_series(X):
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape(n_samples, n_timesteps * n_features)
    return X_flat

# 将展平的数据恢复为3D时间序列格式
def reshape_to_time_series(X_flat, n_timesteps, n_features):
    n_samples = X_flat.shape[0]
    X_3d = X_flat.reshape(n_samples, n_timesteps, n_features)
    return X_3d

# 主函数：对多变量时间序列应用SMOTE
def apply_smote_to_timeseries(X, y, sampling_strategy='auto', random_state=42):
    # 数据标准化（可选，但推荐）
    n_samples, n_timesteps, n_features = X.shape
    scaler = MinMaxScaler()
    
    # 展平数据并标准化
    X_flat = flatten_time_series(X)
    X_flat_scaled = scaler.fit_transform(X_flat)
    
    # 应用SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_smote_flat, y_smote = smote.fit_resample(X_flat_scaled, y)
    
    # 反标准化
    X_smote_flat = scaler.inverse_transform(X_smote_flat)
    
    # 恢复为3D时间序列格式
    X_smote = reshape_to_time_series(X_smote_flat, n_timesteps, n_features)
    
    return X_smote, y_smote

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    X, y = generate_sample_data(n_samples=100, n_timesteps=10, n_features=3)
    print("原始数据形状:", X.shape)
    print("原始标签分布:", np.bincount(y))
    
    # 应用SMOTE增强
    X_augmented, y_augmented = apply_smote_to_timeseries(X, y)
    print("增强后数据形状:", X_augmented.shape)
    print("增强后标签分布:", np.bincount(y_augmented))
    
    # 可视化验证（可选）
    import matplotlib.pyplot as plt
    
    # 绘制第一个样本的第一个特征在时间维度上的变化
    plt.figure(figsize=(12, 6))
    plt.plot(X[0, :, 0], label='Original Sample')
    plt.plot(X_augmented[0, :, 0], label='Augmented Sample')
    plt.legend()
    plt.title('Comparison of Original vs Augmented Time Series')
    plt.show()