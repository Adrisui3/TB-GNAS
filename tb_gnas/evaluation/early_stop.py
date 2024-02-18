import numpy as np


class EarlyStop:
    def __init__(self, patience: int = 1, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.max_val_acc = -np.inf
        self.counter = 0

    def early_stop(self, val_acc: float) -> bool:
        if val_acc > self.max_val_acc:
            self.counter = 0
            self.max_val_acc = val_acc
        elif val_acc + self.min_delta < self.max_val_acc:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
