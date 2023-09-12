import time

import numpy as np
from torch_geometric.datasets import Planetoid

from tbma_gnas.search_strategy.local_search import local_search
from tbma_gnas.search_strategy.simulated_annealing import simulated_annealing

pubmed = Planetoid(root='/tmp/PubMed', name='PubMed')
cora = Planetoid(root='/tmp/Cora', name='Cora')
citeseer = Planetoid(root='/tmp/Citeseer', name='Citeseer')
dfs = [pubmed, cora, citeseer]

for df in dfs:
    res = []
    for _ in range(32):
        print("---- DATASET: ", str(df), " ---- ITER: ", _)
        time_ini = time.time()
        gnn, acc, hist = simulated_annealing(dataset=df, num_iters=150, max_depth=2)
        time_end = time.time()
        runtime = time_end - time_ini
        res.append((gnn, acc, gnn.size(), runtime))
        print("Runtime: ", time.time() - time_ini)
        print("History: ", hist)
        print("Blocks: ", gnn.get_blocks())
        print("Size: ", gnn.size())
        print("Test accuracy:", acc)

    with open("./tbma_gnas/results/sa_trimmed_150_es_models_" + str(df) + ".txt", "a") as f:
        print("Results: ", res, file=f)

    with open("./tbma_gnas/results/sa_trimmed_150_es_summary.txt", "a") as f:
        print("---- DATASET: " + str(df) + "----", file=f)
        print("Best found model: ", max(res, key=lambda x: x[1]), file=f)
        test_accuracies = [x[1] for x in res]
        idx_max = np.argmax(test_accuracies)
        idx_min = np.argmin(test_accuracies)
        sizes = [x[2] for x in res]
        runtimes = [x[3] for x in res]

        acc_corrected = np.delete(test_accuracies, [idx_min, idx_max])
        sizes_corrected = np.delete(sizes, [idx_min, idx_max])
        runtimes_corrected = np.delete(runtimes, [idx_min, idx_max])
        print("Average acc: ", np.mean(acc_corrected), "+/-", np.std(acc_corrected), file=f)
        print("Average size: ", np.mean(sizes_corrected), "+/-", np.std(sizes_corrected), file=f)
        print("Average runtime: ", np.mean(runtimes_corrected), "+/-", np.std(runtimes_corrected), file=f)
