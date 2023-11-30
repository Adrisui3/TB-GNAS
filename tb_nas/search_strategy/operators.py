import random

from tb_nas.search_space.hypermodel import HyperModel
from tb_nas.search_space.search_space import SearchSpace


def increase_depth(space: SearchSpace, model: HyperModel):
    return space.increase_model_depth(model)


def decrease_depth(space: SearchSpace, model: HyperModel):
    return space.reduce_model_depth(model) if len(model.get_blocks()) > 1 else model


def change_layer(space: SearchSpace, model: HyperModel):
    return space.query_for_component(model=model, complete_layer=True)


def change_hyperparameters(space: SearchSpace, model: HyperModel):
    return space.query_for_component(model=model, complete_layer=False)


ALL_OPERATORS = [increase_depth, decrease_depth, change_layer, change_hyperparameters]


def select_operator(weights: list = None):
    op_idx = random.choices(population=range(len(ALL_OPERATORS)), weights=weights, k=1)[0]
    return ALL_OPERATORS[op_idx], op_idx
