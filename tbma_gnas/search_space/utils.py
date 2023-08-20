from .hyperparameters.default_values import DEFAULT_HYPERPARAMETERS
from .hyperparameters.dimension_ratio import DimensionRatio


def has_heads_parameter(layer):
    return "heads" in layer.__dict__.keys()


def has_concat_parameter(layer) -> bool:
    return "concat" in layer.__dict__.keys()


def reset_model_parameters(model_blocks: list):
    for block in model_blocks:
        block[0].reset_parameters()


def get_heads_from_layer(layer) -> int:
    return layer.heads if has_heads_parameter(layer) else 1


def get_concat_from_layer(layer) -> bool:
    return layer.concat if has_concat_parameter(layer) else False


def compute_prev_block_heads(block_idx: int, blocks: list) -> int:
    if block_idx > 0:
        return get_heads_from_layer(blocks[block_idx - 1][0])

    return 1


def compute_prev_block_concat(block_idx: int, blocks: list) -> bool:
    if block_idx > 0:
        return get_concat_from_layer(blocks[block_idx - 1][0])

    return False


def retrieve_layer_config(layer):
    # Given a layer, it retrieves the set of parameters and values which are considered for optimization.
    # It is worth noting that the complete set of parameters of the layer might be bigger.
    prev_params = {key: layer.__dict__[key] for key in layer.__dict__.keys() if
                   key in DEFAULT_HYPERPARAMETERS[layer.__class__.__name__].keys()}

    heads_param = prev_params["heads"] if "heads" in prev_params.keys() else 1
    out_shape = layer.out_channels * heads_param
    if layer.in_channels == out_shape:
        prev_ratio = DimensionRatio.EQUAL
    elif layer.in_channels > out_shape:
        prev_ratio = DimensionRatio.REDUCE
    else:
        prev_ratio = DimensionRatio.INCREASE

    return prev_params, prev_ratio
