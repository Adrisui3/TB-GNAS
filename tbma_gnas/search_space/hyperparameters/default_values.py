from torch_geometric import nn as geom_nn

DEFAULT_HYPERPARAMETERS = {
    geom_nn.GATv2Conv.__name__: {
        "heads": [1, 2, 3, 4, 5, 6, 7, 8],
        "negative_slope": [0.05, 0.1, 0.2, 0.3],
        "dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        "concat": [True, False],
        "fill_value": ["add", "mean", "min", "max", "mul"]
    },

    geom_nn.ChebConv.__name__: {
        "K": [1, 2, 3, 4, 5],
        "normalization": [None, "sym", "rw"]
    },

    geom_nn.TransformerConv.__name__: {
        "heads": [1, 2, 3, 4, 5, 6, 7, 8],
        "dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        "concat": [True, False],
        "beta": [True, False]
    },

    geom_nn.GCNConv.__name__: {
        "improved": [True, False],
        "normalize": [True, False]
    }
}
