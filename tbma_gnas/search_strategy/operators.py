from tbma_gnas.search_space.hypermodel import HyperModel
from tbma_gnas.search_space.search_space import SearchSpace


def increase_depth(space: SearchSpace, model: HyperModel):
    current_depth = len(model.get_blocks())
    return space.query_for_depth(depth=current_depth + 1)


def decrease_depth(space: SearchSpace, model: HyperModel):
    depth = len(model.get_blocks()) - 1
    return space.query_for_depth(depth=depth) if depth >= 1 else model


def change_layer(space: SearchSpace, model: HyperModel):
    return space.query_for_component(model=model, complete_layer=True)


def change_hyperparameters(space: SearchSpace, model: HyperModel):
    return space.query_for_component(model=model, complete_layer=False)
