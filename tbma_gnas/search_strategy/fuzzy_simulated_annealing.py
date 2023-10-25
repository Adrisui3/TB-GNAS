import random

from tbma_gnas.fuzzy_comparator.fuzzy_comparator import accept_optimum, accept_incumbent, redemption
from tbma_gnas.search_space.utils import reset_model_parameters
from tbma_gnas.search_strategy.operators import select_operator, ALL_OPERATORS
from tbma_gnas.search_strategy.utils import setup_search, unhandled_model


def fuzzy_simulated_annealing(dataset, num_iters: int, max_depth: int = None):
    logger, search_space, evaluator, comparator = setup_search(dataset=dataset, max_depth=max_depth, fuzzy=True)
    operator_weights = [1] * len(ALL_OPERATORS)
    model_cache = {}

    logger.info("Generating and training initial model - STARTING")
    best_model, best_val_acc = evaluator.low_fidelity_estimation(model=search_space.get_init_model(), dataset=dataset)
    best_size = best_model.size()
    incumbent_model, incumbent_acc, incumbent_size = best_model, best_val_acc, best_size
    search_space.update_previous_state(model=best_model)
    model_cache[best_model] = best_val_acc
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial model blocks: " + str(best_model.get_blocks()))
    logger.info("Initial validation accuracy: " + str(best_val_acc))
    logger.info("Initial model size: " + str(best_size))

    history = [(-1, best_val_acc, best_size)]
    explored_models = 0
    failed_models = 0

    while explored_models < num_iters:
        logger.info(" --- ITERATION: " + str(explored_models) + " ---")
        operator, op_idx = select_operator(weights=operator_weights)
        logger.info("Selected operator: " + operator.__name__)
        current_model = operator(search_space, incumbent_model)
        logger.info("New model generated: " + str(current_model.get_blocks()))

        try:
            if current_model not in model_cache:
                current_model, current_acc = evaluator.low_fidelity_estimation(model=current_model, dataset=dataset)
                model_cache[current_model] = current_acc
            else:
                logger.info("Cached model, skipping evaluation...")
                current_acc = model_cache[current_model]

            current_size = current_model.size()
            logger.info("Validation accuracy: " + str(current_acc) + " - Size: " + str(current_size))

            acc_label, size_label = comparator.compute_matching_labels(incumbent_size, incumbent_acc, current_size,
                                                                       current_acc)
            logger.info("Fuzzy labels w.r.t incumbent - Accuracy: " + str(acc_label) + " Size: " + str(size_label))

            if accept_incumbent(acc_label=acc_label, size_label=size_label):
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.learn(model=incumbent_model, positive=True)
                search_space.update_previous_state(model=incumbent_model)
                operator_weights[op_idx] += 1
                logger.info("Incumbent updated")
                acc_label_opt, size_label_opt = comparator.compute_matching_labels(best_size, best_val_acc,
                                                                                   incumbent_size,
                                                                                   incumbent_acc)
                logger.info(
                    "Fuzzy labels w.r.t optimum - Accuracy: " + str(acc_label_opt) + " Size: " + str(size_label_opt))
                if accept_optimum(acc_label=acc_label_opt, size_label=size_label_opt):
                    best_model, best_val_acc, best_size = incumbent_model, incumbent_acc, incumbent_size
                    search_space.learn(model=best_model, positive=True)
                    operator_weights[op_idx] += 1
                    history.append((explored_models, best_val_acc, best_size))
                    logger.info("Optimum updated")
            elif redemption(acc_label=acc_label, size_label=size_label) and random.uniform(0, 1) < (
                    num_iters - explored_models) / num_iters:
                logger.info("Incumbent accepted")
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.update_previous_state(model=incumbent_model)

            explored_models += 1
            failed_models = 0

        except Exception as exception:
            unhandled_model(exception, logger, current_model)
            failed_models += 1
            if failed_models > num_iters:
                raise

    logger.info("Evaluating model in test set...")
    reset_model_parameters(best_model.get_blocks())
    best_model, test_acc = evaluator.evaluate_in_test(best_model, dataset)
    logger.info("Test set accuracy: " + str(test_acc))

    return best_model, test_acc, history
