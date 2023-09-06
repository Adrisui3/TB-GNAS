import time

import numpy as np
from torch_geometric.datasets import Planetoid

from tbma_gnas.fuzzy_comparator.fuzzy_comparator import accept_optimum
from tbma_gnas.search_strategy.operators import select_operator, ALL_OPERATORS
from tbma_gnas.search_strategy.utils import setup_search


def local_search(dataset, num_iter: int):
    logger, search_space, evaluator, comparator = setup_search(dataset=dataset)
    model_cache = {}
    operator_weights = [1] * len(ALL_OPERATORS)

    logger.info("Generating and training initial model - STARTING")
    best_model, best_acc = evaluator.low_fidelity_estimation(model=search_space.get_init_model(), dataset=dataset)
    best_size = best_model.size()
    search_space.update_previous_state(model=best_model)
    model_cache[best_model] = best_acc
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial model blocks: " + str(best_model.get_blocks()))
    logger.info("Initial validation accuracy: " + str(best_acc))
    logger.info("Initial model size: " + str(best_size))

    history = [(0, best_acc, best_size)]

    for i in range(num_iter):
        logger.info("Iteration " + str(i))
        operator, op_idx = select_operator(weights=operator_weights)
        logger.info("Selected operator: " + operator.__name__)
        new_model = operator(search_space, best_model)
        logger.info("New model generated: " + str(new_model.get_blocks()))

        try:
            if new_model not in model_cache:
                logger.info("Unvisited model, evaluating...")
                new_model, new_acc = evaluator.low_fidelity_estimation(model=new_model,
                                                                       dataset=dataset)
                model_cache[new_model] = new_acc
            else:
                logger.info("Cached model, skipping evaluation...")
                new_acc = model_cache[new_model]

            new_size = new_model.size()
            logger.info("Validation accuracy: " + str(new_acc) + " - Size: " + str(new_size))

            acc_label, size_label = comparator.compute_matching_labels(best_size, best_acc, new_size, new_acc)
            logger.info("Fuzzy labels - Accuracy: " + str(acc_label) + " Size: " + str(size_label))

            if accept_optimum(acc_label=acc_label, size_label=size_label):
                best_model, best_acc, best_size = new_model, new_acc, new_size
                search_space.learn(model=best_model, positive=True)
                search_space.update_previous_state(model=best_model)
                operator_weights[op_idx] += 1
                history.append((i, best_acc, best_size))
                logger.info("Best model updated")

        except Exception as exception:
            logger.warning("A model could not be handled: " + str(new_model.get_blocks()))
            logger.warning("Size: " + str(new_model.size()))
            if "shapes cannot be multiplied" in str(exception):
                logger.error("Reason: " + str(exception))
                raise
            else:
                logger.warning("Reason: " + str(exception))

    return best_model, best_acc, history


pubmed = Planetoid(root='/tmp/PubMed', name='PubMed')
cora = Planetoid(root='/tmp/Cora', name='Cora')
dfs = [pubmed]

res = []
for df in dfs:
    for _ in range(5):
        print("---- DATASET: ", str(df), " ---- ITER: ", _)
        time_ini = time.time()
        gnn, acc, hist = local_search(dataset=df, num_iter=150)
        res.append((gnn, acc, gnn.size()))
        print("Runtime: ", time.time() - time_ini)
        print("History: ", hist)
        print("Blocks: ", gnn.get_blocks())
        print("Size: ", gnn.size())
        print("Validation accuracy:", acc)

print("Results: ", res)
print("Best found model: ", max(res, key=lambda x: x[1]))
accs = [x[1] for x in res]
sizes = [x[2] for x in res]
print("Average acc: ", np.mean(accs))
print("Average size: ", np.mean(sizes))

'''
trans = geom_nn.TransformerConv(in_channels=50, out_channels=20, heads=3, concat=True)
print(trans.__dict__)
'''

'''
--- PubMed ---
Runtime:  1089.2492578029633
Blocks:  [(GATv2Conv(500, 500, heads=1), Tanh()), (GATv2Conv(500, 3, heads=1), Sigmoid())]
Size:  505012
Validation accuracy: 0.808

'''
