from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class CostSensitiveDataBalancer:
    def __init__(self, classes):
        self.classes = classes

    def compute_weights(self, y):
        class_weights = compute_class_weight('balanced', classes=self.classes, y=y)
        return dict(zip(self.classes, class_weights))