import random

import numpy as np
import torch

from tb_gnas.search_strategy.utils import setup_search

MAX_FAILED_MODELS = 150


def simulated_annealing(dataset, t_ini: float = 3.076e-3, t_end: float = 5.0071e-5, alpha: float = 0.97275,
                        max_depth: int = None):
    logger, search_space, evaluator, _ = setup_search(dataset=dataset, max_depth=max_depth, fuzzy=False)
    model_cache = {}

    logger.info("Generating and training initial model - STARTING")
    best_model, best_val_acc = None, None
    while best_model is None and best_val_acc is None:
        try:
            best_model, best_val_acc = evaluator.low_fidelity_estimation(model=search_space.get_init_model())
        except torch.cuda.OutOfMemoryError:
            logger.warning("Failed to generate initial model. Retrying...")
    best_size = best_model.size()
    incumbent_model, incumbent_acc, incumbent_size = best_model, best_val_acc, best_size
    best_objective = best_val_acc
    incumbent_objective = best_objective
    model_cache[best_model.get_hashable_repr()] = best_val_acc
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
        logger.info(" --- TEMPERATURE: " + str(temp) + " ---")
        logger.info("Allocated VRAM: " + str(torch.cuda.memory_allocated() / 1e9) + " GB")
        current_model = search_space.mutate_hypermodel(incumbent_model)
        logger.info("New model generated: " + str(current_model.get_blocks()))

        try:
            current_repr = current_model.get_hashable_repr()
            if current_repr not in model_cache:
                current_model, current_acc = evaluator.low_fidelity_estimation(model=current_model)
                model_cache[current_repr] = current_acc
            else:
                logger.info("Cached model, skipping evaluation...")
                continue

            current_size = current_model.size()
            current_objective = current_acc
            logger.info("Best model objective function: " + str(best_objective) + " - Size: " + str(best_size))
            logger.info(
                "Incumbent model objective function: " + str(incumbent_objective) + " - Size: " + str(incumbent_size))
            logger.info("New model objective function: " + str(current_objective) + " - Size: " + str(current_size))

            delta = incumbent_objective - current_objective
            if delta < 0:
                incumbent_model, incumbent_objective, incumbent_size = current_model, current_objective, current_size
                incumbent_objective = current_objective
                search_space.learn(model=incumbent_model, positive=True)
                logger.info("Incumbent updated")
                if best_objective < current_objective:
                    best_model, best_objective, best_size = incumbent_model, incumbent_objective, incumbent_size
                    best_objective = current_objective
                    search_space.learn(model=best_model, positive=True)
                    history.append((explored_models, best_objective, best_size))
                    logger.info("Optimum updated")
            elif random.uniform(0, 1) < np.exp(-delta / temp):
                logger.info("Incumbent accepted")
                incumbent_model = current_model
                incumbent_objective = current_objective
                incumbent_size = current_size

            explored_models += 1
            failed_models = 0
            temp = temp * alpha

        except torch.cuda.OutOfMemoryError:
            logger.warning("A model could not be fit in memory: " + str(current_model.get_blocks()))
            failed_models += 1
            if failed_models > MAX_FAILED_MODELS:
                raise

    logger.info("Evaluating model in test set...")
    best_model.reset_parameters()
    best_model, test_acc = evaluator.evaluate_in_test(best_model)
    logger.info("Test set accuracy: " + str(test_acc))

    return best_model, test_acc, history
