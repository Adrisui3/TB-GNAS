import numpy as np
from torch_geometric.datasets import Planetoid

PARAMS_PER_DATASET = {"PubMed":
    {"fuzzy_local_search": {
        "num_iters": 1500,
        "max_depth": 2,
    },
        "fuzzy_simulated_annealing": {
            "num_iters": 1500,
            "max_depth": 2
        },
        "local_search": {
            "num_iters": 1500,
            "max_depth": 2

        },
        "simulated_annealing": {
            "t_ini": 3.076e-3,
            "t_end": 5.0071e-5,
            "alpha": 0.97275,
            "max_depth": 2
        }},
    "Cora":
        {"fuzzy_local_search": {
            "num_iters": 1500,
            "max_depth": 2,
        },
            "fuzzy_simulated_annealing": {
                "num_iters": 1500,
                "max_depth": 2
            },
            "local_search": {
                "num_iters": 1500,
                "max_depth": 2

            },
            "simulated_annealing": {
                "t_ini": 0.0004430253154047865,
                "t_end": 8.345205017383352e-06,
                "alpha": 0.9737,
                "max_depth": 2
            }},
    "Citeseer":
        {"fuzzy_local_search": {
            "num_iters": 1500,
            "max_depth": 2,
        },
            "fuzzy_simulated_annealing": {
                "num_iters": 1500,
                "max_depth": 2
            },
            "local_search": {
                "num_iters": 1500,
                "max_depth": 2

            },
            "simulated_annealing": {
                "t_ini": 0.00017228762265741697,
                "t_end": 2.8373697059103396e-06,
                "alpha": 0.97285,
                "max_depth": 2
            }}
}


def load_datasets():
    pubmed = Planetoid(root='./tb_gnas/experiments/datasets/PubMed', name='PubMed')
    cora = Planetoid(root='./tb_gnas/experiments/datasets/Cora', name='Cora')
    citeseer = Planetoid(root='./tb_gnas/experiments/datasets/Citeseer', name='Citeseer')

    return [pubmed, cora, citeseer]


def trim_results(res):
    test_accuracies = [x[1] for x in res]
    sizes = [x[2] for x in res]
    runtimes = [x[3] for x in res]

    return test_accuracies, sizes, runtimes
