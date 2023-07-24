from enum import Enum
from torch_geometric import nn


class ComponentType(Enum):
    LAYER = {"gatv1": nn.GATConv, "gatv2": nn.GATv2Conv, "gcnconv": nn.GCNConv}
    ACTIVATION = {}
    REGULARIZATION = {}
