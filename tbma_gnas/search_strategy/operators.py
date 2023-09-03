from tbma_gnas.search_space.hypermodel import HyperModel
from tbma_gnas.search_space.search_space import SearchSpace


def increase_depth(space: SearchSpace, model: HyperModel):
    print(model.get_blocks())
    return space.increase_model_depth(model)


def decrease_depth(space: SearchSpace, model: HyperModel):
    print(model.get_blocks())
    return space.reduce_model_depth(model) if len(model.get_blocks()) > 1 else model


def change_layer(space: SearchSpace, model: HyperModel):
    print(model.get_blocks())
    return space.query_for_component(model=model, complete_layer=True)


def change_hyperparameters(space: SearchSpace, model: HyperModel):
    print(model.get_blocks())
    return space.query_for_component(model=model, complete_layer=False)
