from .space_component import LearnableSpaceComponent
from .component_type import ComponentType


class LearnableBlock:
    def __init__(self):
        self.layer = LearnableSpaceComponent(ComponentType.LAYER)
        self.regularization = LearnableSpaceComponent(ComponentType.REGULARIZATION)
        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)

    def learn(self, layer_component: str, regularization_component: str, activation_component: str, positive: bool):
        feedback = 1 if positive else -1
        self.layer.learn(component=layer_component, feedback=feedback)
        self.regularization.learn(component=regularization_component, feedback=feedback)
        self.activation.learn(component=activation_component, feedback=feedback)
