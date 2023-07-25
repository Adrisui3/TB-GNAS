from enum import Enum
from torch_geometric import nn as geom_nn
from torch import nn


class ComponentType(Enum):
    LAYER = {"gatv1": geom_nn.GATConv, "gatv2": geom_nn.GATv2Conv, "gcnconv": geom_nn.GCNConv}
    ACTIVATION = {"relu": nn.ReLU}
    REGULARIZATION = {}
