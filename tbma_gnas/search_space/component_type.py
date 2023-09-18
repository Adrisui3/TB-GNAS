from enum import Enum

from torch import nn
from torch_geometric import nn as geom_nn


class ComponentType(Enum):
    LAYER = {geom_nn.GATv2Conv.__name__: geom_nn.GATv2Conv,
             geom_nn.GATConv.__name__: geom_nn.GATConv,
             geom_nn.ChebConv.__name__: geom_nn.ChebConv,
             geom_nn.GCNConv.__name__: geom_nn.GCNConv,
             geom_nn.GraphConv.__name__: geom_nn.GraphConv,
             geom_nn.TransformerConv.__name__: geom_nn.TransformerConv}

    ACTIVATION = {nn.ReLU.__name__: nn.ReLU,
                  nn.ELU.__name__: nn.ELU,
                  nn.Sigmoid.__name__: nn.Sigmoid,
                  nn.Tanh.__name__: nn.Tanh,
                  nn.Softplus.__name__: nn.Softplus}

    REGULARIZATION = {nn.Dropout.__name__: nn.Dropout}
