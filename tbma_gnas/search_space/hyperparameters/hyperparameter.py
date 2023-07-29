import random
from threading import Lock


class HyperParameter:
    def __init__(self, values: list):
        self.values = values
        self.scores = {value: 1 for value in values}
        self.lock = Lock()

    def learn(self, prev_value, positive: bool):
        feedback = 1 if positive else -1
        with self.lock:
            self.scores[prev_value] = max(self.scores[prev_value] + feedback, 1)

    def query(self):
        with self.lock:
            weights = [self.scores[value] for value in self.values]
            return random.choices(population=self.values, weights=weights, k=1)[0]
