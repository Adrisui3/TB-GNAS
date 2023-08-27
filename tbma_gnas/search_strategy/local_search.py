import random
import time

from torch_geometric.datasets import Planetoid

from tbma_gnas.evaluation.evaluator import Evaluator
from tbma_gnas.fuzzy_comparator.fuzzy_comparator import FuzzyComparator, improvement, penalization
from tbma_gnas.logger.logger import Logger
from tbma_gnas.search_space.search_space import SearchSpace
from tbma_gnas.search_strategy.operators import increase_depth, change_layer, change_hyperparameters, decrease_depth


def local_search(dataset, num_iter: int,
                 operators: list = [increase_depth, change_layer, change_hyperparameters, decrease_depth]):
    logger = Logger()
    logger.info("Starting local search for " + str(num_iter) + " iterations")
    search_space = SearchSpace(num_node_features=dataset.num_node_features, output_shape=dataset.num_classes)
    logger.info("Search Space initialized")
    evaluator = Evaluator()
    logger.info("Evaluator initialized. Device: " + evaluator.get_device())
    comparator = FuzzyComparator()
    logger.info("Fuzzy comparator initialized.")
    model_cache = {}
    operator_weights = [num_iter] * len(operators)

    logger.info("Generating and training initial model - STARTING")
    # TODO: Think of some sort of warmup for the search space
    search_space.query_for_depth(depth=1)
    search_space.query_for_depth(depth=2)
    best_model, best_acc = evaluator.low_fidelity_estimation(model=search_space.query_for_depth(depth=3),
                                                             dataset=dataset)
    best_size = best_model.size()
    search_space.update_previous_state(model=best_model)
    model_cache[best_model] = best_acc
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial model blocks: " + str(best_model.get_blocks()))
    logger.info("Initial validation accuracy: " + str(best_acc))
    logger.info("Initial model size: " + str(best_size))

    history_acc = [best_acc]
    history_size = [best_size]

    for i in range(num_iter):
        logger.info("Iteration " + str(i))
        print(operator_weights)
        op_idx = random.choices(population=range(len(operators)), weights=operator_weights, k=1)[0]
        operator = operators[op_idx]
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

            if improvement(acc_label=acc_label, size_label=size_label):
                best_model = new_model
                best_acc = new_acc
                best_size = new_size
                search_space.learn(model=best_model, positive=True)
                search_space.update_previous_state(model=best_model)
                operator_weights[op_idx] += 1
                history_acc.append(best_acc)
                history_size.append(best_size)
                logger.info("Best model updated")
            elif penalization(acc_label=acc_label, size_label=size_label):
                search_space.learn(model=new_model, positive=False)
                operator_weights[op_idx] = max(operator_weights[op_idx] - 1, 1)
                logger.info("Imposing penalization")

        except Exception as exception:
            logger.warning("A model could not be handled: " + str(new_model.get_blocks()))
            logger.warning("Size: " + str(new_model.size()))
            logger.warning("Reason: " + str(exception))
            operator_weights[op_idx] = max(operator_weights[op_idx] - 1, 1)

    print(history_acc)
    print(history_size)

    return best_model, best_acc


cora = Planetoid(root='/tmp/PubMed', name='PubMed')
t_ini = time.time()
gnn, acc = local_search(dataset=cora, num_iter=500)
print("Runtime: ", time.time() - t_ini)
print("Blocks: ", gnn.get_blocks())
print("Size: ", gnn.size())
print("Validation accuracy:", acc)
