from enum import Enum


class SizeLabel(Enum):
    SMALLER = 1
    EQUAL = 2
    BIGGER = 3


class AccLabel(Enum):
    LESS = 1
    EQUAL = 2
    MORE = 3


def improvement(ref_size: int, ref_val_acc: float, cand_size: int, cand_val_acc: float, acc_tol: float = 0.01,
                size_tol: float = 0.05) -> bool:
    return True
