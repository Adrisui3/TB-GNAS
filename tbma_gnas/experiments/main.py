import time
from datetime import datetime

import numpy as np

from tbma_gnas.experiments.utils import load_datasets, trim_results, PARAMS_PER_DATASET
from tbma_gnas.search_strategy.fuzzy_local_search import fuzzy_local_search
from tbma_gnas.search_strategy.fuzzy_simulated_annealing import fuzzy_simulated_annealing
from tbma_gnas.search_strategy.local_search import local_search
from tbma_gnas.search_strategy.simulated_annealing import simulated_annealing

RESULTS_PATH = "./tbma_gnas/results/"
RUNS = 32

if __name__ == "__main__":
    dfs = load_datasets()

    for alg in [local_search, simulated_annealing, fuzzy_local_search, fuzzy_simulated_annealing]:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%m-%d-%Y_%H:%M:%S")
        for df in dfs:
            res = []
            params = PARAMS_PER_DATASET[df.name][alg.__name__]
            for _ in range(RUNS):
                print("---- DATASET: ", df.name, " ---- ITER: ", _)
                time_ini = time.time()
                gnn, acc, hist = alg(dataset=df, **params)
                time_end = time.time()
                runtime = time_end - time_ini
                res.append((gnn, acc, gnn.size(), runtime, hist))
                print("Runtime: ", runtime)
                print("History: ", hist)
                print("Blocks: ", gnn.get_blocks())
                print("Size: ", gnn.size())
                print("Test accuracy:", acc)

            with open(RESULTS_PATH + "models/" + formatted_datetime + "_" + alg.__name__ + "_" + df.name + ".txt",
                      "a") as f:
                print("Results: ", res, file=f)

            with open(RESULTS_PATH + "summary/" + formatted_datetime + "_" + alg.__name__ + ".txt", "a") as f:
                print("---- DATASET: " + str(df) + "----", file=f)
                print("Best found model: ", max(res, key=lambda x: x[1]), file=f)
                acc_trimmed, sizes_trimmed, runtimes_trimmed = trim_results(res)
                print("Average acc: ", np.mean(acc_trimmed), "+/-", np.std(acc_trimmed), file=f)
                print("Average size: ", np.mean(sizes_trimmed), "+/-", np.std(sizes_trimmed), file=f)
                print("Average runtime: ", np.mean(runtimes_trimmed), "+/-", np.std(runtimes_trimmed), file=f)
