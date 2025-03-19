from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle
from utils.metrics import Check_If_IOH  # 假设已定义

# 生成示例数据（模拟你的场景）
def generate_sample_data(n_samples=100, seq_len=15, n_features=5, pred_len=5):
    X = defaultdict(list)
    for i in range(n_samples):
        X['window_sample_time'].append(np.linspace(0, 14, seq_len))  # 15分钟
        X['Solar8000/NIBP_MBP_window_sample'].append(np.random.randn(seq_len))
        X['prediction_window_time'].append(np.arange(15, 20, 1))  # 5分钟
        X['prediction_maap'].append(np.random.randn(pred_len))
        X['caseid'].append(f"case_{i}")
    prediction_maap = X['prediction_maap']
    y = np.array([Check_If_IOH(seq, IOH_value=65, duration=3) for seq in prediction_maap])
    return X, y

# 随机下采样
def random_undersample(X, y, ratio=1.0):
    """
    X: defaultdict，包含所有特征
    y: 标签数组
    ratio: 多数类保留比例（相对于少数类），默认1.0表示平衡
    """
    # 计算类别分布
    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    n_target_negative = int(n_positive * ratio)  # 目标负类数量

    if n_target_negative >= n_negative:
        return X, y  # 无需下采样

    # 分离正负类索引
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # 随机选择负类子集
    neg_indices_sampled = np.random.choice(neg_indices, n_target_negative, replace=False)
    selected_indices = np.concatenate([pos_indices, neg_indices_sampled])

    # 打乱顺序
    selected_indices = shuffle(selected_indices, random_state=42)

    # 构造下采样后的数据
    X_sampled = defaultdict(list)
    for feature in X.keys():
        feature_data = np.array(X[feature])
        X_sampled[feature] = feature_data[selected_indices].tolist()
    y_sampled = y[selected_indices]

    return X_sampled, y_sampled

# 主流程
if __name__ == "__main__":
    # 生成数据
    X, y = generate_sample_data(n_samples=100)
    print("原始标签分布：", np.bincount(y))

    # 执行下采样
    X_sampled, y_sampled = random_undersample(X, y, ratio=1.0)  # 平衡正负类
    print("下采样后标签分布：", np.bincount(y_sampled))

    # 检查采样后的数据
    print("下采样后样本数：", len(X_sampled['caseid']))
    print("第一个样本的 prediction_maap：", X_sampled['prediction_maap'][0])