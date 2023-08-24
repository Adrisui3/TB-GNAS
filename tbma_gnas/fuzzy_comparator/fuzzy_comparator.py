from enum import Enum

from .fuzzy_variable import FuzzyVariable


class SizeLabel(Enum):
    SMALLER = 1
    EQUAL = 2
    BIGGER = 3


class AccLabel(Enum):
    LESS = 1
    EQUAL = 2
    MORE = 3


class FuzzyComparator:
    ACCURACY_INTERVALS = [-0.05, -0.02, -0.05, -0.02, 0.02, 0.05, 0.02, 0.05]
    SIZE_INTERVALS = [-0.03, -0.015, -0.03, -0.015, 0.015, 0.03, 0.015, 0.03]

    def __init__(self):
        self.acc_variable = FuzzyVariable(labels=[AccLabel.LESS, AccLabel.EQUAL, AccLabel.MORE],
                                          abcdefgh=self.ACCURACY_INTERVALS)
        self.size_variable = FuzzyVariable(labels=[SizeLabel.SMALLER, SizeLabel.EQUAL, SizeLabel.BIGGER],
                                           abcdefgh=self.SIZE_INTERVALS)

    def compute_matching_labels(self, size_ratio: float, acc_ratio: float) -> tuple:
        acc_label = self.acc_variable.compute_matching_label(x=acc_ratio - 1)
        size_label = self.size_variable.compute_matching_label(x=size_ratio - 1)

        print(acc_ratio - 1)
        print(size_ratio - 1)

        return acc_label, size_label

    def improvement(self, ref_size: int, ref_val_acc: float, cand_size: int, cand_val_acc: float) -> bool:
        acc_label, size_label = self.compute_matching_labels(size_ratio=cand_size / ref_size,
                                                             acc_ratio=cand_val_acc / ref_val_acc)

        print(acc_label, size_label)

        return (acc_label == AccLabel.EQUAL and size_label == SizeLabel.SMALLER) or (
                acc_label == AccLabel.EQUAL and size_label == SizeLabel.EQUAL) or (
                acc_label == AccLabel.MORE and size_label == SizeLabel.SMALLER) or (
                acc_label == AccLabel.MORE and size_label == SizeLabel.EQUAL) or (
                acc_label == AccLabel.MORE and size_label == SizeLabel.BIGGER)
