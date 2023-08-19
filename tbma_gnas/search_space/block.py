from typing import Any

from .component_type import ComponentType
from .hyperparameters.dimension_ratio import DimensionRatio
from .hyperparameters.hyperparameters import HyperParameters
from .space_component import LearnableSpaceComponent
from .utils import retrieve_layer_config


class LearnableBlock:
    def __init__(self, is_input: bool = False, is_output: bool = False):
        self.is_input = is_input
        self.is_output = is_output

        self.layer = LearnableSpaceComponent(ComponentType.LAYER)
        self.layer_hyperparameters = HyperParameters()

        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)

        self.prev_layer = None
        self.prev_in_channels = None
        self.prev_out_channels = None
        self.prev_hyperparameters = None
        self.prev_out_channels = None
        self.prev_act = None

    def get_input(self):
        return self.is_input

    def get_output(self):
        return self.is_output

    def disable_output(self):
        self.is_output = False

    def learn(self, layer, activation, positive: bool):
        # Call the learn methods of each component individually with the corresponding feedback
        self.layer.learn(component=layer.__class__.__name__, positive=positive)
        prev_params, prev_ratio = retrieve_layer_config(layer)
        self.layer_hyperparameters.learn_for_layer(layer=layer.__class__.__name__, prev_values=prev_params,
                                                   prev_ratio=prev_ratio, positive=positive)

        self.activation.learn(component=activation.__class__.__name__, positive=positive)

    def rebuild_block(self, new_in_channels: int):
        if self.prev_layer and self.prev_act:
            self.prev_in_channels = new_in_channels
            return self.prev_layer(in_channels=new_in_channels, out_channels=self.prev_out_channels,
                                   **self.prev_hyperparameters), self.prev_act()

    def _fix_heads_output_block(self, sampled_params: dict):
        if "heads" in sampled_params.keys() and self.is_output:
            sampled_params["heads"] = 1

    def _save_new_state(self, init_layer, params: dict, out_channels: int, in_channels: int, init_act):
        self.prev_layer = init_layer
        self.prev_in_channels = in_channels
        self.prev_out_channels = out_channels
        self.prev_hyperparameters = params
        self.prev_act = init_act

    def _compute_out_channels(self, num_node_features: int, output_shape: int, prev_out_channels: int,
                              dim_ratio: DimensionRatio):
        # If the DimensionRatio is set to EQUAL, then the block will keep the input shape and the output shape equals,
        # on the other hand, if it's set to REDUCE, it will reduce the dimension to a half.

        if self.is_output:
            return output_shape

        match dim_ratio:
            case DimensionRatio.EQUAL:
                return prev_out_channels
            case DimensionRatio.REDUCE:
                return max(prev_out_channels // 2, output_shape)
            case DimensionRatio.INCREASE:
                return min(prev_out_channels * 2, num_node_features)

    def query_hyperparameters_for_layer(self, layer) -> tuple[Any, Any]:
        _, new_params = self.layer_hyperparameters.query_for_layer(layer.__class__.__name__)
        self._fix_heads_output_block(new_params)
        self.prev_hyperparameters = new_params
        return self.prev_layer(in_channels=self.prev_in_channels, out_channels=self.prev_out_channels,
                               **new_params), self.prev_act()

    def query(self, prev_out_shape: int, num_node_features: int, output_shape: int) -> tuple[Any, Any]:
        # Query every component individually: layer, parameters and activation function
        init_layer = self.layer.query()
        dim_ratio, params = self.layer_hyperparameters.query_for_layer(init_layer.__name__)
        out_channels = self._compute_out_channels(num_node_features, output_shape, prev_out_shape, dim_ratio)
        init_act = self.activation.query()

        # If the block is an output block, and it has heads parameter, set it to one manually so that the output shape is coherent with the problem's requirements
        self._fix_heads_output_block(params)

        # Save the new state of the block so that it can be rebuilt if required
        self._save_new_state(init_layer, params, out_channels, prev_out_shape, init_act)

        return init_layer(in_channels=prev_out_shape, out_channels=out_channels, **params), init_act()
