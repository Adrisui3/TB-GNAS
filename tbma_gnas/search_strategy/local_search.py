import random

from torch_geometric.datasets import Planetoid

from tbma_gnas.evaluation.evaluator import Evaluator
from tbma_gnas.fuzzy_comparator.fuzzy_comparator import FuzzyComparator
from tbma_gnas.logger.logger import Logger
from tbma_gnas.search_space.search_space import SearchSpace
from tbma_gnas.search_strategy.operators import increase_depth, change_layer, change_hyperparameters


def local_search(dataset, num_iter: int, operators: list = [increase_depth, change_layer, change_hyperparameters]):
    logger = Logger()
    logger.info("Starting local search for " + str(num_iter) + " iterations")
    search_space = SearchSpace(num_node_features=dataset.num_node_features, output_shape=dataset.num_classes)
    logger.info("Search Space initialized")
    evaluator = Evaluator()
    logger.info("Evaluator initialized. Device: " + evaluator.get_device())
    comparator = FuzzyComparator()
    logger.info("Fuzzy comparator initialized.")

    logger.info("Generating and training initial model - STARTING")
    best_model, best_acc = evaluator.low_fidelity_estimation(model=search_space.query_for_depth(depth=1),
                                                             dataset=dataset)
    best_size = best_model.size()
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial validation accuracy: " + str(best_acc))
    logger.info("Initial model size: " + str(best_size))

    for i in range(num_iter):
        logger.info("Iteration " + str(i))
        operator = random.choice(operators)
        logger.info("Selected operator: " + operator.__name__)
        new_model, new_acc = evaluator.low_fidelity_estimation(model=operator(search_space, best_model),
                                                               dataset=dataset)
        new_size = new_model.size()
        logger.info("New model generated. Validation accuracy: " + str(new_acc) + ". Size: " + str(new_size))

        if comparator.improvement(best_size, best_acc, new_size, new_acc):
            best_model = new_model
            best_acc = new_acc
            best_size = new_size
            search_space.learn(model=best_model, positive=True)
            logger.info("New best model found. Validation accuracy: " + str(best_acc) + ". Size: " + str(best_size))
        else:
            search_space.learn(model=new_model, positive=False)

    return best_model, best_acc


cora = Planetoid(root='/tmp/Cora', name='Cora')
gnn, acc = local_search(dataset=cora, num_iter=100)
print(gnn.get_blocks())
print(gnn.size())
print(acc)
