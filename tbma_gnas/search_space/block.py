import random
from typing import Any

from .component_type import ComponentType
from .hyperparameters.dimension_ratio import DimensionRatio
from .hyperparameters.hyperparameters import HyperParameters
from .space_component import LearnableSpaceComponent
from .utils import retrieve_layer_config, get_module_params


def compute_out_channels(is_output: bool, num_node_features: int, data_out_shape: int, prev_out_channels: int,
                         dim_ratio: DimensionRatio, params: dict) -> int:
    # If the DimensionRatio is set to EQUAL, then the block will keep the input shape and the output shape equals,
    # on the other hand, if it's set to REDUCE, it will reduce the dimension to a half.
    if is_output:
        return data_out_shape

    concat = params["concat"] if "concat" in params.keys() else False
    heads = params["heads"] if "heads" in params.keys() and concat else 1
    match dim_ratio:
        case DimensionRatio.EQUAL:
            return max(prev_out_channels // heads, data_out_shape)
        case DimensionRatio.REDUCE:
            # return max(prev_out_channels // (2 * heads), data_out_shape)
            return random.randint(data_out_shape, max(prev_out_channels // 2, data_out_shape))
        case DimensionRatio.INCREASE:
            # return min((prev_out_channels * 2) // heads, num_node_features)
            return random.randint(prev_out_channels + 1, 2 * prev_out_channels)


def fix_heads_output_block(is_output: bool, sampled_params: dict):
    concat = sampled_params["concat"] if "concat" in sampled_params.keys() else False
    heads = sampled_params["heads"] if "heads" in sampled_params.keys() else 1
    if concat and "concat" in sampled_params.keys() and heads > 1 and is_output:
        sampled_params["concat"] = False


class LearnableBlock:
    def __init__(self, is_input: bool = False, is_output: bool = False):
        self.is_input = is_input
        self.is_output = is_output

        self.layer = LearnableSpaceComponent(ComponentType.LAYER)
        self.layer_hyperparameters = HyperParameters()

        self.regularization = LearnableSpaceComponent(ComponentType.REGULARIZATION)
        self.regularization_hyperparameters = HyperParameters()

        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)
        self.activation_hyperparameters = HyperParameters()

        self.prev_layer = None
        self.prev_in_channels = None
        self.prev_out_channels = None
        self.prev_layer_hyperparameters = None

        self.prev_act = None
        self.prev_act_hyperparameters = None

        self.prev_reg = None
        self.prev_reg_hyperparameters = None

    def get_input(self):
        return self.is_input

    def get_output(self):
        return self.is_output

    def disable_output(self):
        self.is_output = False

    def learn(self, layer, regularization, activation, positive: bool):
        # Call the learn methods of each component individually with the corresponding feedback
        self.layer.learn(component=layer.__class__.__name__, positive=positive)
        prev_params, prev_ratio = retrieve_layer_config(layer)
        self.layer_hyperparameters.learn_for_layer(layer=layer.__class__.__name__, prev_values=prev_params,
                                                   prev_ratio=prev_ratio, positive=positive)

        self.regularization.learn(component=regularization.__class__.__name__, positive=positive)
        reg_params = get_module_params(regularization)
        self.regularization_hyperparameters.learn_for_module(regularization.__class__.__name__, reg_params, positive)

        self.activation.learn(component=activation.__class__.__name__, positive=positive)

    def set_previous_state(self, block: tuple):
        self.prev_layer = self.layer.get_components()[block[0].__class__.__name__]
        self.prev_in_channels = block[0].in_channels
        self.prev_out_channels = block[0].out_channels
        prev_hyperparams, _ = retrieve_layer_config(block[0])
        self.prev_layer_hyperparameters = prev_hyperparams

        self.prev_act = self.activation.get_components()[block[1].__class__.__name__]

        self.prev_reg = self.regularization.get_components()[block[2].__class__.__name__]
        self.prev_reg_hyperparameters = get_module_params(block[2])

    def rebuild_block(self, new_in_channels: int = None, new_out_channels: int = None, new_params: dict = None) -> \
            tuple[Any, Any, Any]:
        in_channels = new_in_channels if new_in_channels else self.prev_in_channels
        out_channels = new_out_channels if new_out_channels else self.prev_out_channels
        params = new_params if new_params else self.prev_layer_hyperparameters
        return self.prev_layer(in_channels=in_channels, out_channels=out_channels,
                               **params), self.prev_act(), self.prev_reg(**self.prev_reg_hyperparameters)

    def query_hyperparameters_for_block(self, block) -> tuple[Any, Any, Any]:
        _, new_params = self.layer_hyperparameters.query_for_layer(block[0].__class__.__name__)
        fix_heads_output_block(self.is_output, new_params)
        new_reg_params = self.regularization_hyperparameters.query_for_module(block[2].__class__.__name__)
        return self.prev_layer(in_channels=self.prev_in_channels, out_channels=self.prev_out_channels,
                               **new_params), self.prev_act(), self.prev_reg(**new_reg_params)

    def query_dimension_ratio_for_layer(self, layer) -> DimensionRatio:
        dim_ratio, _ = self.layer_hyperparameters.query_for_layer(layer.__class__.__name__)
        return dim_ratio

    def query(self, prev_out_shape: int, num_node_features: int, data_out_shape: int) -> tuple[Any, Any, Any]:
        # Query every component individually: layer, parameters and activation function
        init_layer = self.layer.query()
        dim_ratio, params = self.layer_hyperparameters.query_for_layer(init_layer.__name__)

        # If the block is an output block, and it has heads parameter, set it to one manually so that the output shape is coherent with the problem's requirements
        fix_heads_output_block(self.is_output, params)

        # Compute the output shape
        out_channels = compute_out_channels(self.is_output, num_node_features, data_out_shape, prev_out_shape,
                                            dim_ratio,
                                            params)

        init_act = self.activation.query()

        init_reg = self.regularization.query()
        reg_params = self.regularization_hyperparameters.query_for_module(init_reg.__name__)

        return init_layer(in_channels=prev_out_shape, out_channels=out_channels, **params), init_act(), init_reg(
            **reg_params)
