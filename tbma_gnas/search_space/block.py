from .component_type import ComponentType
from .hyperparameters.default_values import DEFAULT_HYPERPARAMETERS
from .hyperparameters.dimension_ratio import DimensionRatio
from .hyperparameters.hyperparameters import HyperParameters
from .space_component import LearnableSpaceComponent


class LearnableBlock:
    def __init__(self, is_input: bool = False, is_output: bool = False):
        self.is_input = is_input
        self.is_output = is_output

        self.layer = LearnableSpaceComponent(ComponentType.LAYER)
        self.layer_hyperparameters = HyperParameters()

        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)

    def get_input(self):
        return self.is_input

    def disable_output(self):
        self.is_output = False

    def learn(self, layer, activation, positive: bool):
        self.layer.learn(component=layer.__class__.__name__, positive=positive)

        prev_params = {key: layer.__dict__[key] for key in layer.__dict__.keys() if
                       key in DEFAULT_HYPERPARAMETERS[layer.__class__.__name__].keys()}
        prev_ratio = DimensionRatio.EQUAL if layer.in_channels == layer.out_channels else DimensionRatio.REDUCE
        self.layer_hyperparameters.learn_for_layer(layer=layer.__class__.__name__, prev_values=prev_params,
                                                   prev_ratio=prev_ratio, positive=positive)

        self.activation.learn(component=activation.__class__.__name__, positive=positive)

    def query(self, prev_out_channels: int, output_shape: int):
        init_layer = self.layer.query()
        dim_ratio, params = self.layer_hyperparameters.query_for_layer(init_layer.__name__)
        out_channels = output_shape if self.is_output else (
            prev_out_channels if dim_ratio == DimensionRatio.EQUAL else max(prev_out_channels // 2, output_shape))

        init_act = self.activation.query()

        return init_layer(in_channels=prev_out_channels, out_channels=out_channels, **params), init_act()
