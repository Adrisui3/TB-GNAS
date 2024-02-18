from typing import Any, List

import numpy as np
import skfuzzy as fuzz


class FuzzyVariable:
    def __init__(self, labels: list, abcdefgh: list):
        self.labels = labels
        self.config = [[-np.inf, -np.inf, abcdefgh[0], abcdefgh[1]],
                       [abcdefgh[2], abcdefgh[3], abcdefgh[4], abcdefgh[5]],
                       [abcdefgh[6], abcdefgh[7], abcdefgh[8], abcdefgh[9]],
                       [abcdefgh[10], abcdefgh[11], abcdefgh[12], abcdefgh[13]],
                       [abcdefgh[14], abcdefgh[15], np.inf, np.inf]]

    def compute_matching_labels(self, x: float) -> list[Any]:
        memberships = []
        for label, params in zip(self.labels, self.config):
            member = fuzz.trapmf(x=np.asarray([x]), abcd=params)
            memberships.append((label, member[0]))

        return list(filter(lambda tup: tup[1] > 0, memberships))
