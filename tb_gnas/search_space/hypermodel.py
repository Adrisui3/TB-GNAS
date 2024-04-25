import torch
import torch.nn as nn
import torch.nn.functional as F

from tb_gnas.search_space.pyg_gnn_layer import GeoLayer


def act_map(act):
    if act == "linear":
        return torch.nn.Identity()
    elif act == "elu":
        return torch.nn.ELU()
    elif act == "sigmoid":
        return torch.nn.Sigmoid()
    elif act == "tanh":
        return torch.nn.Tanh()
    elif act == "relu":
        return torch.nn.ReLU()
    elif act == "relu6":
        return torch.nn.ReLU6()
    elif act == "softplus":
        return torch.nn.Softplus()
    elif act == "leaky_relu":
        return torch.nn.LeakyReLU()
    else:
        raise Exception("wrong activate function")


class HyperModel(torch.nn.Module):
    def __init__(self, model_blocks: list):
        super().__init__()
        self.blocks = model_blocks
        self.layers = nn.ModuleList()
        for block in model_blocks:
            self.layers.append(
                GeoLayer(in_channels=block["in_channels"], out_channels=block["out_channels"], heads=block["heads"],
                         concat=block["concat"], dropout=block["dropout"], att_type=block["attention"],
                         agg_type=block["aggregator"]))
            self.layers.append(act_map(block["activation"]))

    def get_blocks(self):
        return self.blocks

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers) - 1, 2):
            x = self.layers[i](x, edge_index)
            x = self.layers[i + 1](x)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_hashable_repr(self):
        return tuple([tuple(block.items()) for block in self.blocks])
