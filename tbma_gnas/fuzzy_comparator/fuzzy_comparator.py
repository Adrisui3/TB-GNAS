from enum import Enum

from .fuzzy_variable import FuzzyVariable


class SizeLabel(Enum):
    MUCH_SMALLER = 1
    SMALLER = 2
    EQUAL = 3
    BIGGER = 4
    MUCH_BIGGER = 6


class AccLabel(Enum):
    MUCH_LESS = 1
    LESS = 2
    EQUAL = 3
    MORE = 4
    MUCH_MORE = 5


def penalization(acc_label: AccLabel, size_label: SizeLabel) -> bool:
    return (acc_label == AccLabel.EQUAL and size_label == SizeLabel.MUCH_BIGGER) or (
            acc_label == AccLabel.MUCH_LESS and size_label == SizeLabel.EQUAL) or (
            acc_label == AccLabel.MUCH_LESS and size_label == SizeLabel.BIGGER) or (
            acc_label == AccLabel.MUCH_LESS and size_label == SizeLabel.MUCH_BIGGER) or (
            acc_label == AccLabel.LESS and size_label == SizeLabel.BIGGER) or (
            acc_label == AccLabel.LESS and size_label == SizeLabel.MUCH_BIGGER)


def accept_optimum(acc_label: AccLabel, size_label: SizeLabel) -> bool:
    return (acc_label == AccLabel.EQUAL and size_label == SizeLabel.MUCH_SMALLER) or (
            acc_label == AccLabel.EQUAL and size_label == SizeLabel.SMALLER) or (
            acc_label == AccLabel.MORE and size_label == SizeLabel.MUCH_SMALLER) or acc_label == AccLabel.MORE or acc_label == AccLabel.MUCH_MORE


def accept_incumbent(acc_label: AccLabel, size_label: SizeLabel) -> bool:
    return accept_optimum(acc_label, size_label) or (acc_label == AccLabel.EQUAL and (
            (size_label == SizeLabel.BIGGER) or size_label == SizeLabel.MUCH_BIGGER)) or (
            acc_label == AccLabel.MORE and size_label == SizeLabel.MUCH_BIGGER)


class FuzzyComparator:
    ACCURACY_INTERVALS = [-0.1, -0.05, -0.1, -0.05, -0.02, 0.00, -0.01, 0.00, 0.00, 0.01, 0.00, 0.03, 0.05, 0.1, 0.05,
                          0.1]
    SIZE_INTERVALS = [-0.15, -0.1, -0.15, -0.1, -0.05, 0.00, -0.025, 0.00, 0.00, 0.015, 0.00, 0.025, 0.05, 0.1, 0.05,
                      0.1]

    def __init__(self):
        self.acc_variable = FuzzyVariable(
            labels=[AccLabel.MUCH_LESS, AccLabel.LESS, AccLabel.EQUAL, AccLabel.MORE, AccLabel.MUCH_MORE],
            abcdefgh=self.ACCURACY_INTERVALS)
        self.size_variable = FuzzyVariable(
            labels=[SizeLabel.MUCH_SMALLER, SizeLabel.SMALLER, SizeLabel.EQUAL, SizeLabel.BIGGER,
                    SizeLabel.MUCH_BIGGER],
            abcdefgh=self.SIZE_INTERVALS)

    def compute_matching_labels(self, ref_size: int, ref_val_acc: float, cand_size: int, cand_val_acc: float) -> tuple:
        acc_label = self.acc_variable.compute_matching_label(x=cand_val_acc - ref_val_acc)
        size_label = self.size_variable.compute_matching_label(x=(cand_size / ref_size) - 1)

        return acc_label, size_label
