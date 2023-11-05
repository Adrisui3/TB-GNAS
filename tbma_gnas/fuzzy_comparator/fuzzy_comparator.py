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


class RuleConsequent(Enum):
    NEW_BEST = 1
    NEW_INCUMBENT = 2
    REDEMPTION = 3
    REJECT = 4


class FuzzyComparator:
    ACCURACY_LABELS = [-0.1, -0.05, -0.1, -0.05, -0.02, 0.00, -0.01, 0.00, 0.00, 0.01, 0.00, 0.03, 0.05, 0.1, 0.05, 0.1]

    SIZE_LABELS = [-0.15, -0.1, -0.15, -0.1, -0.05, 0.00, -0.025, 0.00, 0.00, 0.015, 0.00, 0.025, 0.05, 0.1, 0.05, 0.1]

    RULE_SET = {
        AccLabel.MUCH_LESS: {
            SizeLabel.MUCH_SMALLER: RuleConsequent.REJECT,
            SizeLabel.SMALLER: RuleConsequent.REJECT,
            SizeLabel.EQUAL: RuleConsequent.REJECT,
            SizeLabel.BIGGER: RuleConsequent.REJECT,
            SizeLabel.MUCH_BIGGER: RuleConsequent.REJECT
        },

        AccLabel.LESS: {
            SizeLabel.MUCH_SMALLER: RuleConsequent.REDEMPTION,
            SizeLabel.SMALLER: RuleConsequent.REDEMPTION,
            SizeLabel.EQUAL: RuleConsequent.REDEMPTION,
            SizeLabel.BIGGER: RuleConsequent.REJECT,
            SizeLabel.MUCH_BIGGER: RuleConsequent.REJECT
        },

        AccLabel.EQUAL: {
            SizeLabel.MUCH_SMALLER: RuleConsequent.NEW_BEST,
            SizeLabel.SMALLER: RuleConsequent.NEW_BEST,
            SizeLabel.EQUAL: RuleConsequent.NEW_INCUMBENT,
            SizeLabel.BIGGER: RuleConsequent.NEW_INCUMBENT,
            SizeLabel.MUCH_BIGGER: RuleConsequent.NEW_INCUMBENT
        },

        AccLabel.MORE: {
            SizeLabel.MUCH_SMALLER: RuleConsequent.NEW_BEST,
            SizeLabel.SMALLER: RuleConsequent.NEW_BEST,
            SizeLabel.EQUAL: RuleConsequent.NEW_BEST,
            SizeLabel.BIGGER: RuleConsequent.NEW_BEST,
            SizeLabel.MUCH_BIGGER: RuleConsequent.NEW_BEST
        },

        AccLabel.MUCH_MORE: {
            SizeLabel.MUCH_SMALLER: RuleConsequent.NEW_BEST,
            SizeLabel.SMALLER: RuleConsequent.NEW_BEST,
            SizeLabel.EQUAL: RuleConsequent.NEW_BEST,
            SizeLabel.BIGGER: RuleConsequent.NEW_BEST,
            SizeLabel.MUCH_BIGGER: RuleConsequent.NEW_BEST
        }
    }

    def __init__(self):
        self.acc_variable = FuzzyVariable(
            labels=[AccLabel.MUCH_LESS, AccLabel.LESS, AccLabel.EQUAL, AccLabel.MORE, AccLabel.MUCH_MORE],
            abcdefgh=self.ACCURACY_LABELS)
        self.size_variable = FuzzyVariable(
            labels=[SizeLabel.MUCH_SMALLER, SizeLabel.SMALLER, SizeLabel.EQUAL, SizeLabel.BIGGER,
                    SizeLabel.MUCH_BIGGER],
            abcdefgh=self.SIZE_LABELS)

    def compute_fired_rules(self, ref_size: int, ref_val_acc: float, cand_size: int, cand_val_acc: float) -> RuleConsequent:
        acc_labels = self.acc_variable.compute_matching_labels(x=cand_val_acc - ref_val_acc)
        size_labels = self.size_variable.compute_matching_labels(x=(cand_size / ref_size) - 1)

        max_acc_label = max(acc_labels, key=lambda tup: tup[1])[0]
        max_size_label = max(size_labels, key=lambda tup: tup[1])[0]

        return self.RULE_SET[max_acc_label][max_size_label]