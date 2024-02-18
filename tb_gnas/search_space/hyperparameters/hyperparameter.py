import random


class HyperParameter:
    def __init__(self, values: list):
        self.values = values
        self.scores = {value: 1 for value in values}

    def learn(self, prev_value, positive: bool):
        feedback = 1 if positive else -1
        self.scores[prev_value] = max(self.scores[prev_value] + feedback, 1)

    def query(self):
        weights = [self.scores[value] for value in self.values]
        return random.choices(population=self.values, weights=weights, k=1)[0]
