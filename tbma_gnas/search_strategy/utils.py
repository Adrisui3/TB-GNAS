from tbma_gnas.evaluation.evaluator import Evaluator
from tbma_gnas.fuzzy_comparator.fuzzy_comparator import FuzzyComparator
from tbma_gnas.logger.logger import Logger
from tbma_gnas.search_space.hypermodel import HyperModel
from tbma_gnas.search_space.search_space import SearchSpace


def setup_search(dataset, ):
    logger = Logger()
    search_space = SearchSpace(num_node_features=dataset.num_node_features, data_out_shape=dataset.num_classes)
    logger.info("Search Space initialized")
    evaluator = Evaluator()
    logger.info("Evaluator initialized. Device: " + evaluator.get_device())
    comparator = FuzzyComparator()
    logger.info("Fuzzy comparator initialized.")

    return logger, search_space, evaluator, comparator


def unhandled_model(exception: Exception, logger: Logger, model: HyperModel):
    logger.warning("A model could not be handled: " + str(model.get_blocks()))
    logger.warning("Size: " + str(model.size()))
    if "shapes cannot be multiplied" in str(exception):
        logger.error("Reason: " + str(exception))
        raise
    else:
        logger.warning("Reason: " + str(exception))
