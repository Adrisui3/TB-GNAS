import random

import numpy as np

from tbma_gnas.fuzzy_comparator.fuzzy_comparator import accept_optimum
from tbma_gnas.search_strategy.operators import select_operator, ALL_OPERATORS
from tbma_gnas.search_strategy.utils import setup_search, unhandled_model


def simulated_annealing(dataset, t_ini: float, t_end: float, alpha: float):
    logger, search_space, evaluator, comparator = setup_search(dataset=dataset)
    operator_weights = [1] * len(ALL_OPERATORS)
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
        logger.info(" --- TEMPERATURE: " + str(temp) + "---")
        operator, op_idx = select_operator(weights=operator_weights)
        logger.info("Selected operator: " + operator.__name__)
        current_model = operator(search_space, incumbent_model)
        logger.info("New model generated: " + str(current_model.get_blocks()))

        try:
            if current_model not in model_cache:
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
            logger.info("Fuzzy labels w.r.t incumbent - Accuracy: " + str(acc_label) + " Size: " + str(size_label))

            delta_acc = incumbent_acc - current_acc
            delta_size = current_size - incumbent_size
            logger.info("Deltas - " + "Validation acc: " + str(delta_acc) + " - Size: " + str(delta_size))
            if delta_acc < 0 and delta_size <= 0:
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.learn(model=incumbent_model, positive=True)
                search_space.update_previous_state(model=incumbent_model)
                operator_weights[op_idx] += 1
                logger.info("Incumbent updated")
                acc_label_opt, size_label_opt = comparator.compute_matching_labels(best_size, best_acc, incumbent_size,
                                                                                   incumbent_acc)
                logger.info(
                    "Fuzzy labels w.r.t optimum - Accuracy: " + str(acc_label_opt) + " Size: " + str(size_label_opt))
                if accept_optimum(acc_label=acc_label_opt, size_label=size_label_opt):
                    best_model, best_acc, best_size = incumbent_model, incumbent_acc, incumbent_size
                    search_space.learn(model=best_model, positive=True)
                    operator_weights[op_idx] += 1
                    history.append((temp, best_acc, best_size))
                    logger.info("Optimum updated")
            elif random.uniform(0.01, 0.99) < np.exp(-delta_acc / temp) and delta_size < 0:
                logger.info("Incumbent accepted")
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.update_previous_state(model=incumbent_model)
            else:
                operator_weights[op_idx] = max(operator_weights[op_idx] - 1, 1)

        except Exception as exception:
            unhandled_model(exception, logger, current_model)

        temp = temp * alpha

    return best_model, best_acc, history
