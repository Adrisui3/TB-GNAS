import random

import numpy as np

from tb_gnas.search_space.utils import reset_model_parameters
from tb_gnas.search_strategy.operators import select_operator, ALL_OPERATORS
from tb_gnas.search_strategy.utils import setup_search, unhandled_model, objective_function

MAX_FAILED_MODELS = 150


def simulated_annealing(dataset, t_ini: float = 3.076e-3, t_end: float = 5.0071e-5, alpha: float = 0.97275,
                        max_depth: int = None):
    logger, search_space, evaluator, _ = setup_search(dataset=dataset, max_depth=max_depth, fuzzy=False)
    operator_weights = [1] * len(ALL_OPERATORS)
    model_cache = {}

    logger.info("Generating and training initial model - STARTING")
    best_model, best_val_acc = evaluator.low_fidelity_estimation(model=search_space.get_init_model())
    best_size = best_model.size()
    incumbent_model, incumbent_acc, incumbent_size = best_model, best_val_acc, best_size
    best_objective = objective_function(best_val_acc, best_size)
    incumbent_objective = best_objective
    search_space.update_previous_state(model=best_model)
    model_cache[best_model] = best_val_acc
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial model blocks: " + str(best_model.get_blocks()))
    logger.info("Initial validation accuracy: " + str(best_val_acc))
    logger.info("Initial model size: " + str(best_size))

    history = [(-1, best_val_acc, best_size)]
    explored_models = 0
    failed_models = 0

    temp = t_ini
    while temp > t_end:
        logger.info(" --- ITERATION: " + str(explored_models) + " ---")
        operator, op_idx = select_operator(weights=operator_weights)
        logger.info("Selected operator: " + operator.__name__)
        current_model = operator(search_space, incumbent_model)
        logger.info("New model generated: " + str(current_model.get_blocks()))

        try:
            if current_model not in model_cache:
                current_model, current_acc = evaluator.low_fidelity_estimation(model=current_model)
                model_cache[current_model] = current_acc
            else:
                logger.info("Cached model, skipping evaluation...")
                current_acc = model_cache[current_model]

            current_size = current_model.size()
            logger.info("Validation accuracy: " + str(current_acc) + " - Size: " + str(current_size))

            current_objective = objective_function(current_acc, current_size)
            logger.info("Best model objective function: " + str(best_objective))
            logger.info("Incumbent model objective function: " + str(incumbent_objective))
            logger.info("New model objective function: " + str(current_objective))

            delta = incumbent_objective - current_objective
            if delta < 0:
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                incumbent_objective = current_objective
                search_space.learn(model=incumbent_model, positive=True)
                search_space.update_previous_state(model=incumbent_model)
                operator_weights[op_idx] += 1
                logger.info("Incumbent updated")
                if best_objective < current_objective:
                    best_model, best_val_acc, best_size = incumbent_model, incumbent_acc, incumbent_size
                    best_objective = current_objective
                    search_space.learn(model=best_model, positive=True)
                    operator_weights[op_idx] += 1
                    history.append((explored_models, best_val_acc, best_size))
                    logger.info("Optimum updated")
            elif random.uniform(0, 1) < np.exp(-delta / temp):
                logger.info("Incumbent accepted")
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.update_previous_state(model=incumbent_model)

            explored_models += 1
            failed_models = 0
            temp = temp * alpha

        except Exception as exception:
            unhandled_model(exception, logger, current_model)
            failed_models += 1
            if failed_models > MAX_FAILED_MODELS:
                raise

    logger.info("Evaluating model in test set...")
    reset_model_parameters(best_model.get_blocks())
    best_model, test_acc = evaluator.evaluate_in_test(best_model)
    logger.info("Test set accuracy: " + str(test_acc))

    return best_model, test_acc, history
