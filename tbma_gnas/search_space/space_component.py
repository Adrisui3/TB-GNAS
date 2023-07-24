import random
from component_type import ComponentType
from threading import Lock


class LearnableSpaceComponent:

    def __init__(self, type):
        self.type = type
        self.components = type.value
        self.scores = {component_str: 1 for component_str in self.components.keys()}
        self.lock = Lock()

    def learn(self, component, feedback):
        with self.lock:
            if self.scores[component] + feedback >= 1:
                self.scores[component] += feedback
            else:
                self.scores[component] = 1

    def query(self):
        options = list(self.components.keys())
        with self.lock:
            weights = [self.scores[component] for component in options]
            return self.components[random.choices(population=options, weights=weights, k=1)[0]]


layer_test = LearnableSpaceComponent(type=ComponentType.LAYER)
for _ in range(3):
    print(layer_test.query())
