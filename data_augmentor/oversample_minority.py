import numpy as np
from collections import defaultdict
from data_augmentor.augm_basic import IOHDataAugmentor

class OversampleMinorityDataBalancer(IOHDataAugmentor):
    def __init__(self, args, X: defaultdict, method='oversample_minority', oversample_factor=8):
        super(OversampleMinorityDataBalancer, self).__init__(args, X, method)
        self.oversample_factor = oversample_factor

    def augment(self, X):
        self.y = self._get_ioh_label(IOH_value=65, stime=self.args.exp_stime)
        print("原始标签分布:", np.bincount(self.y))
        
        # 计算少数类样本的比例并调整 oversample_factor
        # minority_count = np.sum(self.y == 1)
        # majority_count = np.sum(self.y == 0)
        # if minority_count > 0:
        #     self.oversample_factor = max(1, majority_count // minority_count)  # 根据比例调整
        # print("调整后的 oversample_factor:", self.oversample_factor)
        
        print("oversample_factor:", self.oversample_factor)
        
        minority_indices = np.where(self.y == 1)[0]  # 低血压样本索引
        X_augmented = defaultdict(list)
        y_augmented = []

        # 保留所有原始样本
        for feature in X.keys():
            feature_data = np.array(X[feature])
            X_augmented[feature].extend(feature_data)
        y_augmented.extend(self.y)

        # 复制少数类样本
        for _ in range(self.oversample_factor - 1):  # 已包含原始样本，故减 1
            for idx in minority_indices:
                for feature in X.keys():
                    X_augmented[feature].append(X[feature][idx])
                y_augmented.append(1)

        print("增强后标签分布:", np.bincount(y_augmented))
        return X_augmented, np.array(y_augmented)
