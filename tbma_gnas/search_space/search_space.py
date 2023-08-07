import copy
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

    def learn_for_depth(self, model: HyperModel, positive: bool):
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

            return HyperModel(model)
