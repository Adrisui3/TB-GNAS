import random

from tbma_gnas.fuzzy_comparator.fuzzy_comparator import RuleConsequent
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

            rule_consequent_incumbent = comparator.compute_fired_rules(incumbent_size, incumbent_acc, current_size,current_acc)
            logger.info("Fired rule w.r.t incumbent - " + str(rule_consequent_incumbent))

            if rule_consequent_incumbent[0] == RuleConsequent.NEW_INCUMBENT or rule_consequent_incumbent[0] == RuleConsequent.NEW_BEST:
                incumbent_model, incumbent_acc, incumbent_size = current_model, current_acc, current_size
                search_space.learn(model=incumbent_model, positive=True)
                search_space.update_previous_state(model=incumbent_model)
                operator_weights[op_idx] += 1
                logger.info("Incumbent updated")
                rule_consequent_optimum = comparator.compute_fired_rules(best_size, best_val_acc, incumbent_size,incumbent_acc)
                logger.info("Fired rule w.r.t optimum - " + str(rule_consequent_optimum))
                if rule_consequent_optimum[0] == RuleConsequent.NEW_BEST:
                    best_model, best_val_acc, best_size = incumbent_model, incumbent_acc, incumbent_size
                    search_space.learn(model=best_model, positive=True)
                    operator_weights[op_idx] += 1
                    history.append((explored_models, best_val_acc, best_size))
                    logger.info("Optimum updated")
            elif rule_consequent_incumbent[0] == RuleConsequent.REDEMPTION and random.uniform(0, 1) < (num_iters - explored_models) / num_iters:
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
