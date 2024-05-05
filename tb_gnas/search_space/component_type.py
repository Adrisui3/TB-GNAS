from enum import Enum


class ComponentType(Enum):
    ATTENTION = {'gat', 'gcn', 'cos', 'const', 'gat_sym', 'linear', 'generalized_linear'}
    AGGREGATOR = {'sum', 'mean', 'max'}
    ACTIVATION = {'sigmoid', 'tanh', 'relu', 'linear', 'softplus', 'leaky_relu', 'relu6', 'elu'}
    CONCAT = [True]
    HEADS = {1, 2, 4, 6, 8, 16}
    HIDDEN_UNITS = {16, 32, 64, 128, 256, 512}
    DROPOUT = {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
