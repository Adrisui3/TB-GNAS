import copy

from .hypermodel import HyperModel
from .learnable_block import LearnableBlock
from .utils import get_heads_from_layer, get_concat_from_layer


def compute_prev_out_shape(prev_layer):
    prev_out_shape = prev_layer.out_channels
    concat = get_concat_from_layer(prev_layer)
    if concat:
        heads = get_heads_from_layer(prev_layer)
        prev_out_shape *= heads

    return prev_out_shape


class SearchSpace:
    def __init__(self, num_node_features: int, data_out_shape: int, max_depth: int):
        self.data_out_shape = data_out_shape
        self.num_node_features = num_node_features
        self.max_depth = max_depth
        self.space = {1: [LearnableBlock(is_input=True, is_output=True)]}

    def learn(self, model: HyperModel, positive: bool):
        depth = len(model.get_blocks())
        for mod_block, block in zip(model.get_blocks(), self.space[depth]):
            block.learn(layer=mod_block[0], regularization=mod_block[2], activation=mod_block[1], positive=positive)

    def query_for_depth(self, depth: int) -> HyperModel:
        model = []
        # If this is the first time the search space has been queried for a model of such depth, initialize it.
        # It is worth noting that the queries must be in ascending order and there cannot be any gaps; that is:
        # If the space has depths 1, 2 and 3, before querying for 5, a query for 4 must happen.
        self._extend_search_space(depth)

        # Iterate over the blocks and query them with the appropriate input and output dimensions
        for block in self.space[depth]:
            prev_out_shape = compute_prev_out_shape(model[-1]) if not block.get_input() else self.num_node_features
            gen_block = block.query(prev_out_shape=prev_out_shape, data_out_shape=self.data_out_shape)
            model.append(gen_block)

        return HyperModel(model_blocks=model)

    def _extend_search_space(self, depth: int):
        if depth not in self.space.keys():
            self.space[depth] = copy.deepcopy(self.space[depth - 1])
            self.space[depth][-1].disable_output()
            self.space[depth].append(LearnableBlock(is_output=True))