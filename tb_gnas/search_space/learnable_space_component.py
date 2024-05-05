import random

from .component_type import ComponentType


class LearnableSpaceComponent:

    def __init__(self, component_type: ComponentType):
        self.type = component_type
        self.components = component_type.value
        self.scores = {component: 1 for component in self.components}

    def get_components(self) -> set:
        return self.components

    def get_scores(self) -> dict:
        return self.scores

    def learn(self, component, positive: bool):
        feedback = 1 if positive else -1
        self.scores[component] = max(self.scores[component] + feedback, 1)

    def query(self, prev_value=None):
        options = list(self.components)
        if prev_value is not None:
            options.remove(prev_value)

        weights = [self.scores[component] for component in options]
        return random.choices(population=options, weights=weights, k=1)[0]
