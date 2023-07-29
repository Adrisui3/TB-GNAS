from torch_geometric import nn

from component_type import ComponentType
from space_component import LearnableSpaceComponent


class LearnableBlock:
    def __init__(self, is_input: bool = False, is_output: bool = False):
        self.is_input = is_input
        self.is_output = is_output
        self.layer = LearnableSpaceComponent(ComponentType.LAYER)
        self.regularization = LearnableSpaceComponent(ComponentType.REGULARIZATION)
        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)

    def learn(self, layer_component: str, regularization_component: str, activation_component: str, positive: bool):
        self.layer.learn(component=layer_component, positive=positive)
        self.regularization.learn(component=regularization_component, positive=positive)
        self.activation.learn(component=activation_component, positive=positive)

    def set_as_input(self):
        self.is_input = True

    def is_input(self):
        return self.is_input

    def set_as_output(self):
        self.is_output = True

    def is_output(self):
        return self.is_output

    def query(self):
        init_layer = self.layer.query()
        init_reg = self.regularization.query()
        init_act = self.activation.query()

        return init_layer, init_reg, init_act


params = {"heads": 5}
test = nn.GATv2Conv(in_channels=-1, out_channels=5, **params)
print(type(dir(test)))

print(str(nn.GATv2Conv.__name__))
