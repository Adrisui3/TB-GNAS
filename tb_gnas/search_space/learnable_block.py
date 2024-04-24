import random
from typing import Any

import torch

from .component_type import ComponentType
from .hyperparameters.dimension_ratio import DimensionRatio
from .learnable_space_component import LearnableSpaceComponent
from .pyg_gnn_layer import GeoLayer
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

    def query(self, prev_out_shape: int, data_out_shape: int) -> GeoLayer:
        att_type = self.attention.query()
        agg_type = self.aggregator.query()
        heads = self.heads.query()
        dropout = self.dropout.query()
        concat = self.concat.query() if not self.is_output else False
        hidden_units = self.hidden_units.query() if not self.is_output else data_out_shape
        activation = self.activation.query()

        return GeoLayer(in_channels=prev_out_shape, out_channels=hidden_units, heads=heads, concat=concat,
                        dropout=dropout, att_type=att_type, agg_type=agg_type, act_type=activation)
