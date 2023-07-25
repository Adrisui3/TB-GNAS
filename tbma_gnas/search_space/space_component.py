import random
from threading import Lock
from .component_type import ComponentType


class LearnableSpaceComponent:

    def __init__(self, component_type: ComponentType):
        self.type = component_type
        self.components = component_type.value
        self.scores = {component_str: 1 for component_str in self.components.keys()}
        self.lock = Lock()

    def get_components(self) -> dict:
        return self.components

    def get_scores(self) -> dict:
        return self.scores

    def learn(self, component: str, feedback: int):
        with self.lock:
            if component in self.scores:
                if self.scores[component] + feedback >= 1:
                    self.scores[component] += feedback
                else:
                    self.scores[component] = 1

    def query(self):
        options = list(self.components.keys())
        with self.lock:
            weights = [self.scores[component] for component in options]
            return self.components[random.choices(population=options, weights=weights, k=1)[0]]
