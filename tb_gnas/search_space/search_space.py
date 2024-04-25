import copy
import random

from .hypermodel import HyperModel
from .learnable_block import LearnableBlock


def compute_output_dimensions(prev_config):
    prev_out_shape = prev_config["out_channels"]
    concat = prev_config["concat"]
    if concat:
        heads = prev_config["heads"]
        prev_out_shape *= heads

    return prev_out_shape


class SearchSpace:
    def __init__(self, num_node_features: int, data_out_shape: int, max_depth: int):
        self.data_out_shape = data_out_shape
        self.num_node_features = num_node_features
        self.max_depth = max_depth
        self.space = {1: [LearnableBlock(is_input=True, is_output=True)]}

    def get_init_model(self):
        return self.query_for_depth(depth=2)

    def learn(self, model: HyperModel, positive: bool):
        depth = len(model.get_blocks())
        for block_config, block in zip(model.get_blocks(), self.space[depth]):
            block.learn(block_config, positive)

    def query_for_depth(self, depth: int) -> HyperModel:
        model = []
        # If this is the first time the search space has been queried for a model of such depth, initialize it.
        # It is worth noting that the queries must be in ascending order and there cannot be any gaps; that is:
        # If the space has depths 1, 2 and 3, before querying for 5, a query for 4 must happen.
        self._extend_search_space(depth)

        # Iterate over the blocks and query them with the appropriate input and output dimensions
        for block in self.space[depth]:
            prev_out_shape = compute_output_dimensions(model[-1]) if not block.get_input() else self.num_node_features
            gen_block = block.query(prev_out_shape=prev_out_shape, data_out_shape=self.data_out_shape)
            model.append(gen_block)

        return HyperModel(model_blocks=model)

    def _extend_search_space(self, depth: int):
        if depth not in self.space.keys():
            self.space[depth] = copy.deepcopy(self.space[depth - 1])
            self.space[depth][-1].disable_output()
            self.space[depth].append(LearnableBlock(is_output=True))

    def mutate_hypermodel(self, model: HyperModel):
        blocks = model.get_blocks()
        depth = len(model.get_blocks())
        block_idx = random.choice(range(depth))

        block_config = blocks[block_idx]
        prev_out_dimensions = compute_output_dimensions(block_config)
        blocks[block_idx] = self.space[depth][block_idx].mutate_block(block_config)
        mutated_out_dimensions = compute_output_dimensions(blocks[block_idx])

        if mutated_out_dimensions != prev_out_dimensions:
            blocks[block_idx + 1]["in_channels"] = mutated_out_dimensions

        return HyperModel(model_blocks=blocks)
