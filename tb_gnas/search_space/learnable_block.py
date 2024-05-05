import copy
import random

from .component_type import ComponentType
from .hyperparameters.dimension_ratio import DimensionRatio
from .learnable_space_component import LearnableSpaceComponent


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

    def learn(self, config: dict, positive: bool):
        # Call the learn methods of each component individually with the corresponding feedback
        self.attention.learn(config["attention"], positive)
        self.aggregator.learn(config["aggregator"], positive)
        self.activation.learn(config["activation"], positive)
        # self.concat.learn(config["concat"], positive)
        self.heads.learn(config["heads"], positive)
        if not self.is_output:
            self.hidden_units.learn(config["out_channels"], positive)
        self.dropout.learn(config["dropout"], positive)

    def query(self, prev_out_shape: int, data_out_shape: int) -> dict:
        att_type = self.attention.query()
        agg_type = self.aggregator.query()
        heads = self.heads.query()
        dropout = self.dropout.query()
        concat = self.concat.query() if not self.is_output else False
        hidden_units = self.hidden_units.query() if not self.is_output else data_out_shape
        activation = self.activation.query()

        return {"in_channels": prev_out_shape, "out_channels": hidden_units, "heads": heads, "concat": concat,
                "dropout": dropout, "attention": att_type, "aggregator": agg_type, "activation": activation}

    def query_element(self, element: str, prev_value=None):
        match element:
            case "attention":
                return self.attention.query(prev_value)
            case "aggregator":
                return self.aggregator.query(prev_value)
            case "activation":
                return self.activation.query(prev_value)
            case "out_channels":
                return self.hidden_units.query(prev_value)
            case "concat":
                return self.concat.query(prev_value)
            case "dropout":
                return self.dropout.query(prev_value)
            case "heads":
                return self.heads.query(prev_value)

    def mutate_block(self, original_config: dict) -> dict:
        mutated_block = copy.deepcopy(original_config)

        candidates = list(mutated_block.keys())
        candidates.remove("in_channels")
        candidates.remove("concat")
        if self.is_output:
            candidates.remove("out_channels")

        param_to_mutate = random.choice(candidates)
        mutated_block[param_to_mutate] = self.query_element(param_to_mutate, mutated_block[param_to_mutate])
        return mutated_block
