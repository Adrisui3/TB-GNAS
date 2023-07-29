from enum import Enum

from torch import nn
from torch_geometric import nn as geom_nn


class ComponentType(Enum):
    LAYER = {geom_nn.GATConv.__name__: geom_nn.GATConv,
             geom_nn.GATv2Conv.__name__: geom_nn.GATv2Conv,
             geom_nn.GCNConv.__name__: geom_nn.GCNConv}

    ACTIVATION = {nn.ReLU.__name__: nn.ReLU}

    REGULARIZATION = {}
