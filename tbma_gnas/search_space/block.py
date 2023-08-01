from .component_type import ComponentType
from .hyperparameters.hyperparameters import HyperParameters
from .space_component import LearnableSpaceComponent


class LearnableBlock:
    def __init__(self, is_input: bool = False, is_output: bool = False):
        self.in_channels = 1
        self.out_channels = 1

        self.is_input = is_input
        self.is_output = is_output

        self.layer = LearnableSpaceComponent(ComponentType.LAYER)
        self.layer_hyperparameters = HyperParameters()

        self.regularization = LearnableSpaceComponent(ComponentType.REGULARIZATION)

        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)

    def learn(self, layer_component: str, regularization_component: str, activation_component: str, positive: bool):
        self.layer.learn(component=layer_component, positive=positive)
        self.regularization.learn(component=regularization_component, positive=positive)
        self.activation.learn(component=activation_component, positive=positive)

    def query(self):
        init_layer = self.layer.query()
        dim_ratio, params = self.layer_hyperparameters.query_for_layer(init_layer.__name__)

        init_reg = self.regularization.query()
        init_act = self.activation.query()

        return init_layer, init_reg, init_act
