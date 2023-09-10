import time

import numpy as np
from torch_geometric.datasets import Planetoid

from tbma_gnas.search_strategy.local_search import local_search

pubmed = Planetoid(root='/tmp/PubMed', name='PubMed')
cora = Planetoid(root='/tmp/Cora', name='Cora')
citeseer = Planetoid(root='/tmp/Citeseer', name='Citeseer')
dfs = [pubmed, cora, citeseer]

for df in dfs:
    res = []
    for _ in range(30):
        print("---- DATASET: ", str(df), " ---- ITER: ", _)
        time_ini = time.time()
        gnn, acc, hist = local_search(dataset=df, num_iters=150, max_depth=2)
        time_end = time.time()
        runtime = time_end - time_ini
        res.append((gnn, acc, gnn.size(), runtime))
        print("Runtime: ", time.time() - time_ini)
        print("History: ", hist)
        print("Blocks: ", gnn.get_blocks())
        print("Size: ", gnn.size())
        print("Test accuracy:", acc)

    with open("./tbma_gnas/results/sa_30_baseline_models_" + str(df) + ".txt", "a") as f:
        print("Results: ", res, file=f)

    with open("./tbma_gnas/results/sa_30_baseline_summary.txt", "a") as f:
        print("---- DATASET: " + str(df) + "----", file=f)
        print("Best found model: ", max(res, key=lambda x: x[1]), file=f)
        accs = [x[1] for x in res]
        sizes = [x[2] for x in res]
        runtimes = [x[3] for x in res]
        print("Average acc: ", np.mean(accs), file=f)
        print("Average size: ", np.mean(sizes), file=f)
        print("Average runtime: ", np.mean(runtimes), file=f)
