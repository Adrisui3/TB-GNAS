from tb_gnas.evaluation.evaluator import Evaluator
from tb_gnas.fuzzy_comparator.fuzzy_comparator import FuzzyComparator
from tb_gnas.logger.logger import Logger
from tb_gnas.search_space.hypermodel import HyperModel
from tb_gnas.search_space.search_space import SearchSpace


def setup_search(dataset, max_depth: int, fuzzy: bool):
    logger = Logger()
    search_space = SearchSpace(num_node_features=dataset.num_node_features, data_out_shape=dataset.num_classes, max_depth=max_depth)
    logger.info("Search Space initialized")
    evaluator = Evaluator(logger, dataset)
    logger.info("Evaluator initialized. Device: " + evaluator.get_device())

    comparator = None
    if fuzzy:
        comparator = FuzzyComparator()
        logger.info("Fuzzy comparator initialized.")

    return logger, search_space, evaluator, comparator


def objective_function(acc: float, size: int) -> float:
    return acc / size


def unhandled_model(exception: Exception, logger: Logger, model: HyperModel):
    logger.warning("A model could not be handled: " + str(model.get_blocks()))
    logger.warning("Size: " + str(model.size()))
    if "shapes cannot be multiplied" in str(exception):
        logger.error("Reason: " + str(exception))
        raise
    else:
        logger.warning("Reason: " + str(exception))
