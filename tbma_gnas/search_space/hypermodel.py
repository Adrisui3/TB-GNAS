import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import retrieve_layer_config, get_module_params


class HyperModel(torch.nn.Module):
    def __init__(self, model_blocks: list):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer, activation, regularization in model_blocks:
            self.layers.append(layer)
            self.layers.append(activation)
            self.layers.append(regularization)

    def get_blocks(self):
        return [(self.layers[i], self.layers[i + 1], self.layers[i + 2]) for i in range(0, len(self.layers) - 2, 3)]

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers) - 2, 3):
            x = self.layers[i](x, edge_index)
            x = self.layers[i + 1](x)
            x = self.layers[i + 2](x)

        return F.log_softmax(x, dim=1)

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __hash__(self):
        to_hash = []
        for bl in self.get_blocks():
            config, _ = retrieve_layer_config(bl[0])
            config["in_channels"] = bl[0].in_channels
            config["out_channels"] = bl[0].out_channels

            reg_config = get_module_params(bl[2])

            to_hash.append(tuple([bl[0].__class__.__name__, tuple(sorted(config.items())), bl[1].__class__.__name__,
                                  bl[2].__class__.__name__, tuple(sorted(reg_config.items()))]))

        to_hash = tuple(to_hash)
        return hash(frozenset(to_hash))
