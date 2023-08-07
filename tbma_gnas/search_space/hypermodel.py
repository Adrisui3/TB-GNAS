import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperModel(torch.nn.Module):
    def __init__(self, model_blocks: list):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer, activation in model_blocks:
            self.layers.append(layer)
            self.layers.append(activation)

    def get_blocks(self):
        return [(self.layers[i], self.layers[i + 1]) for i in range(0, len(self.layers) - 1, 2)]

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers) - 1, 2):
            x = self.layers[i](x, edge_index)
            if self.layers[i + 1]:
                x = self.layers[i + 1](x)

        return F.log_softmax(x, dim=1)
