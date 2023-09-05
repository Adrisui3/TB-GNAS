import random
import time

import numpy as np
from torch_geometric.datasets import Planetoid

from tbma_gnas.evaluation.evaluator import Evaluator
from tbma_gnas.fuzzy_comparator.fuzzy_comparator import FuzzyComparator, improvement, penalization
from tbma_gnas.logger.logger import Logger
from tbma_gnas.search_space.search_space import SearchSpace
from tbma_gnas.search_strategy.operators import increase_depth, change_layer, decrease_depth, change_hyperparameters


def simulated_annealing(dataset, t_ini: float, t_end: float, alpha: float,
                        operators: list = [increase_depth, change_layer, change_hyperparameters, decrease_depth]):
    logger = Logger()
    logger.info("Starting simulated annealing with " + str(t_ini) + " initial temperature and " + str(
        t_end) + " final temperature.")
    search_space = SearchSpace(num_node_features=dataset.num_node_features, data_out_shape=dataset.num_classes)
    logger.info("Search Space initialized")
    evaluator = Evaluator()
    logger.info("Evaluator initialized. Device: " + evaluator.get_device())
    comparator = FuzzyComparator()
    logger.info("Fuzzy comparator initialized.")
    operator_weights = [1] * len(operators)
    model_cache = {}

    logger.info("Generating and training initial model - STARTING")
    best_model, best_acc = evaluator.low_fidelity_estimation(model=search_space.get_init_model(), dataset=dataset)
    best_size = best_model.size()
    incumbent_model, incumbent_acc, incumbent_size = best_model, best_acc, best_size
    search_space.update_previous_state(model=best_model)
    model_cache[best_model] = best_acc
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial model blocks: " + str(best_model.get_blocks()))
    logger.info("Initial validation accuracy: " + str(best_acc))
    logger.info("Initial model size: " + str(best_size))

    history = [(t_ini, best_acc, best_size)]
    temp = t_ini
    while temp > t_end:
        logger.info("Temperature " + str(temp))
        op_idx = random.choices(population=range(len(operators)), weights=operator_weights, k=1)[0]
        operator = operators[op_idx]
        logger.info("Selected operator: " + operator.__name__)
        current_model = operator(search_space, incumbent_model)
        logger.info("New model generated: " + str(current_model.get_blocks()))

        try:
            if current_model not in model_cache:
                logger.info("Unvisited model, evaluating...")
                current_model, current_acc = evaluator.low_fidelity_estimation(model=current_model,
                                                                               dataset=dataset)
                model_cache[current_model] = current_acc
            else:
                logger.info("Cached model, skipping evaluation...")
                current_acc = model_cache[current_model]

            current_size = current_model.size()
            logger.info("Validation accuracy: " + str(current_acc) + " - Size: " + str(current_size))

            acc_label, size_label = comparator.compute_matching_labels(incumbent_size, incumbent_acc, current_size,
                                                                       current_acc)
            logger.info("Fuzzy labels - Accuracy: " + str(acc_label) + " Size: " + str(size_label))

            delta_acc = incumbent_acc - current_acc
            logger.info("Validation accuracy delta: " + str(delta_acc))
            if improvement(acc_label=acc_label, size_label=size_label):
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.learn(model=incumbent_model, positive=True)
                search_space.update_previous_state(model=incumbent_model)
                operator_weights[op_idx] += 1
                logger.info("Incumbent updated")
                if incumbent_acc >= best_acc:
                    best_model, best_acc, best_size = incumbent_model, incumbent_acc, incumbent_size
                    search_space.learn(model=best_model, positive=True)
                    operator_weights[op_idx] += 1
                    history.append((temp, best_acc, best_size))
                    logger.info("Best model updated")
            elif delta_acc < 0 and random.uniform(0, 1) < np.exp(-delta_acc / temp):
                logger.info("Incumbent accepted")
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.update_previous_state(model=incumbent_model)
            elif penalization(acc_label=acc_label, size_label=size_label):
                search_space.learn(model=current_model, positive=False)
                operator_weights[op_idx] = max(operator_weights[op_idx] - 1, 1)
                logger.info("Imposing penalization")

        except Exception as exception:
            logger.warning("A model could not be handled: " + str(current_model.get_blocks()))
            logger.warning("Size: " + str(current_model.size()))
            if "shapes cannot be multiplied" in str(exception):
                logger.error("Reason: " + str(exception))
                raise
            else:
                logger.warning("Reason: " + str(exception))

        temp = temp * alpha

    return best_model, best_acc, history


pubmed = Planetoid(root='/tmp/PubMed', name='PubMed')
cora = Planetoid(root='/tmp/Cora', name='Cora')
dfs = [pubmed]

for df in dfs:
    for _ in range(1):
        print("---- DATASET: ", str(df), " ---- ITER: ", _)
        t_ini = time.time()
        gnn, acc, hist = simulated_annealing(dataset=df, t_ini=4.92, t_end=0.033, alpha=0.995)
        print("Runtime: ", time.time() - t_ini)
        print("History: ", hist)
        print("Blocks: ", gnn.get_blocks())
        print("Size: ", gnn.size())
        print("Validation accuracy:", acc)
