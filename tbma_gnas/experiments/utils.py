import numpy as np
from torch_geometric.datasets import Planetoid


def load_datasets():
    pubmed = Planetoid(root='./tbma_gnas/experiments/datasets/PubMed', name='PubMed')
    cora = Planetoid(root='./tbma_gnas/experiments/datasets/Cora', name='Cora')
    citeseer = Planetoid(root='./tbma_gnas/experiments/datasets/Citeseer', name='Citeseer')

    return [pubmed, cora]


def trim_results(res):
    test_accuracies = [x[1] for x in res]
    sizes = [x[2] for x in res]
    runtimes = [x[3] for x in res]

    idx_max = np.argmax(test_accuracies)
    idx_min = np.argmin(test_accuracies)

    acc_trimmed = np.delete(test_accuracies, [idx_min, idx_max])
    sizes_trimmed = np.delete(sizes, [idx_min, idx_max])
    runtimes_trimmed = np.delete(runtimes, [idx_min, idx_max])

    return acc_trimmed, sizes_trimmed, runtimes_trimmed
