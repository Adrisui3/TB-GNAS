import random

from .component_type import ComponentType


class LearnableSpaceComponent:

    def __init__(self, component_type: ComponentType):
        self.type = component_type
        self.components = component_type.value
        self.scores = {component_str: 1 for component_str in self.components.keys()}

    def get_components(self) -> dict:
        return self.components

    def get_scores(self) -> dict:
        return self.scores

    def learn(self, component: str, positive: bool):
        feedback = 1 if positive else -1
        self.scores[component] = max(self.scores[component] + feedback, 1)

    def query(self):
        options = list(self.components.keys())
        weights = [self.scores[component] for component in options]
        return self.components[random.choices(population=options, weights=None, k=1)[0]]
