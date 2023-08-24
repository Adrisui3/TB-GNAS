import numpy as np
import skfuzzy as fuzz


class FuzzyVariable:
    def __init__(self, labels: list, abcdefgh: list):
        self.labels = labels
        self.config = [[-np.inf, -np.inf, abcdefgh[0], abcdefgh[1]],
                       [abcdefgh[2], abcdefgh[3], abcdefgh[4], abcdefgh[5]],
                       [abcdefgh[6], abcdefgh[7], np.inf, np.inf]]

    def compute_matching_label(self, x: float):
        memberships = []
        for label, params in zip(self.labels, self.config):
            member = fuzz.trapmf(x=np.asarray([x]), abcd=params)
            memberships.append((label, member[0]))

        return max(memberships, key=lambda tup: tup[1])[0]
