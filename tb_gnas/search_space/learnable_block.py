import random
from typing import Any

from .component_type import ComponentType
from .hyperparameters.dimension_ratio import DimensionRatio
from .learnable_space_component import LearnableSpaceComponent
from .utils import retrieve_layer_config, get_module_params


def compute_out_channels(is_output: bool, data_out_shape: int, prev_out_channels: int, dim_ratio: DimensionRatio,
                         params: dict) -> int:
    if is_output:
        return data_out_shape

    concat = params["concat"] if "concat" in params.keys() else False
    heads = params["heads"] if "heads" in params.keys() and concat else 1
    match dim_ratio:
        case DimensionRatio.EQUAL:
            return max(prev_out_channels // heads, data_out_shape)
        case DimensionRatio.REDUCE:
            return random.randint(data_out_shape, max(prev_out_channels // 2, data_out_shape))
        case DimensionRatio.INCREASE:
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

        self.attention = LearnableSpaceComponent(ComponentType.ATTENTION)
        self.aggregator = LearnableSpaceComponent(ComponentType.AGGREGATOR)
        self.activation = LearnableSpaceComponent(ComponentType.ACTIVATION)
        self.concat = LearnableSpaceComponent(ComponentType.CONCAT)
        self.heads = LearnableSpaceComponent(ComponentType.HEADS)
        self.hidden_units = LearnableSpaceComponent(ComponentType.HIDDEN_UNITS)
        self.dropout = LearnableSpaceComponent(ComponentType.DROPOUT)

    def get_input(self):
        return self.is_input

    def get_output(self):
        return self.is_output

    def disable_output(self):
        self.is_output = False

    def learn(self, att: str, aggr: str, act: str, heads: int, hidden_units: int, dropout: float, concat: bool,
              positive: bool):
        # Call the learn methods of each component individually with the corresponding feedback
        self.attention.learn(att, positive)
        self.aggregator.learn(aggr, positive)
        self.activation.learn(act, positive)
        self.concat.learn(concat, positive)
        self.heads.learn(heads, positive)
        self.hidden_units.learn(hidden_units, positive)
        self.dropout.learn(dropout, positive)

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

    def query_hyperparameters_for_block(self, block, data_out_shape: int, prev_block_out_shape: int) -> tuple[
        Any, Any, Any]:
        new_dim_ratio, new_params = self.layer_hyperparameters.query_for_layer(block[0].__class__.__name__)
        fix_heads_output_block(self.is_output, new_params)
        new_reg_params = self.regularization_hyperparameters.query_for_module(block[2].__class__.__name__)
        new_out_channels = compute_out_channels(self.is_output, data_out_shape, prev_block_out_shape, new_dim_ratio,
                                                new_params)

        return self.prev_layer(in_channels=self.prev_in_channels, out_channels=new_out_channels,
                               **new_params), self.prev_act(), self.prev_reg(
            **new_reg_params)

    def query_dimension_ratio_for_layer(self, layer) -> DimensionRatio:
        dim_ratio, _ = self.layer_hyperparameters.query_for_layer(layer.__class__.__name__)
        return dim_ratio

    def query(self, prev_out_shape: int, data_out_shape: int) -> tuple[Any, Any, Any]:
        # Query every component and its hyperparameters individually
        init_layer = self.layer.query()
        dim_ratio, params = self.layer_hyperparameters.query_for_layer(init_layer.__name__)
        init_act = self.activation.query()
        init_reg = self.regularization.query()
        reg_params = self.regularization_hyperparameters.query_for_module(init_reg.__name__)

        # If the block is an output block, and it has heads parameter, set it to one manually so that the output shape is coherent with the problem's requirements
        fix_heads_output_block(self.is_output, params)

        # Compute the output shape
        out_channels = compute_out_channels(self.is_output, data_out_shape, prev_out_shape, dim_ratio, params)

        return init_layer(in_channels=prev_out_shape, out_channels=out_channels, **params), init_act(), init_reg(
            **reg_params)
