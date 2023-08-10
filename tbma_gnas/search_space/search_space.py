import copy
import random
from threading import Lock

from .block import LearnableBlock
from .hypermodel import HyperModel


def has_heads_parameter(layer):
    return "heads" in layer.__dict__.keys()


class SearchSpace:
    def __init__(self, num_node_features: int, output_shape: int):
        self.output_shape = output_shape
        self.num_node_features = num_node_features
        self.space = {1: [LearnableBlock(is_input=True, is_output=True)]}
        self.lock = Lock()

    def learn(self, model: HyperModel, positive: bool):
        with self.lock:
            depth = len(model.get_blocks())
            for lay_act, block in zip(model.get_blocks(), self.space[depth]):
                block.learn(layer=lay_act[0], activation=lay_act[1], positive=positive)

    def query_model_for_depth(self, depth: int) -> HyperModel:
        with self.lock:
            model = []
            if depth not in self.space.keys():
                self.space[depth] = copy.deepcopy(self.space[depth - 1])
                self.space[depth][-1].disable_output()
                self.space[depth].append(LearnableBlock(is_output=True))

            for block in self.space[depth]:
                heads = 1
                if model:
                    heads = model[-1][0].heads if not block.get_input() and has_heads_parameter(model[-1][0]) else heads
                prev_channels = model[-1][0].out_channels if not block.get_input() else self.num_node_features

                gen_block = block.query(prev_out_channels=heads * prev_channels, output_shape=self.output_shape)
                model.append(gen_block)

            return HyperModel(model_blocks=model)

    def query_single_layer(self, model: HyperModel) -> HyperModel:
        blocks = model.get_blocks()

        print(blocks)

        model_depth = len(blocks)
        block_idx = random.randint(0, model_depth - 1)
        print("Index:", block_idx)

        old_block_heads = blocks[block_idx][0].heads if has_heads_parameter(blocks[block_idx][0]) else 1
        old_block_out_shape = old_block_heads * blocks[block_idx][0].out_channels
        print("Old output shape:", old_block_out_shape)

        prev_block_heads = 1
        if block_idx > 0:
            prev_block_heads = blocks[block_idx - 1][0].heads if has_heads_parameter(
                blocks[block_idx - 1][0]) else prev_block_heads
        print("Previous block heads:", prev_block_heads)

        new_in_channels = self.num_node_features if block_idx == 0 else prev_block_heads * blocks[block_idx - 1][
            0].out_channels
        new_block = self.space[model_depth][block_idx].query(prev_out_channels=new_in_channels,
                                                             output_shape=self.output_shape)
        blocks[block_idx] = new_block

        new_block_heads = new_block[0].heads if has_heads_parameter(new_block[0]) else 1
        new_block_out_shape = new_block_heads * new_block[0].out_channels
        print("New output shape:", new_block_out_shape)
        if new_block_out_shape != old_block_out_shape and block_idx != model_depth - 1:
            print("Adjusting next block")
            blocks[block_idx + 1] = self.space[model_depth][block_idx + 1].rebuild_block(
                new_in_channels=new_block_heads * blocks[block_idx][0].out_channels)

        print(blocks)

        return HyperModel(model_blocks=blocks)


space = SearchSpace(num_node_features=25, output_shape=5)
_ = space.query_model_for_depth(depth=2)
_ = space.query_model_for_depth(depth=3)
test_model = space.query_model_for_depth(depth=4)
layer_changed = space.query_single_layer(test_model)
