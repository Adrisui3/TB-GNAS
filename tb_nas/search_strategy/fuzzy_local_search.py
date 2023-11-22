from tbma_gnas.fuzzy_comparator.fuzzy_comparator import RuleConsequent
from tbma_gnas.search_space.utils import reset_model_parameters
from tbma_gnas.search_strategy.operators import select_operator, ALL_OPERATORS
from tbma_gnas.search_strategy.utils import setup_search, unhandled_model


def fuzzy_local_search(dataset, num_iters: int, max_depth: int = None):
    logger, search_space, evaluator, comparator = setup_search(dataset=dataset, max_depth=max_depth, fuzzy=True)
    model_cache = {}
    operator_weights = [num_iters] * len(ALL_OPERATORS)

    logger.info("Generating and training initial model - STARTING")
    best_model, best_acc = evaluator.low_fidelity_estimation(model=search_space.get_init_model())
    best_size = best_model.size()
    search_space.update_previous_state(model=best_model)
    model_cache[best_model] = best_acc
    logger.info("Generating and training initial model - DONE")
    logger.info("Initial model blocks: " + str(best_model.get_blocks()))
    logger.info("Initial validation accuracy: " + str(best_acc))
    logger.info("Initial model size: " + str(best_size))

    history = [(0, best_acc, best_size)]
    explored_models = 0
    failed_models = 0

    while explored_models < num_iters:
        logger.info(" --- ITERATION: " + str(explored_models) + " ---")
        operator, op_idx = select_operator(weights=operator_weights)
        logger.info("Selected operator: " + operator.__name__)
        new_model = operator(search_space, best_model)
        logger.info("New model generated: " + str(new_model.get_blocks()))

        try:
            if new_model not in model_cache:
                logger.info("Unvisited model, evaluating...")
                new_model, new_acc = evaluator.low_fidelity_estimation(model=new_model)
                model_cache[new_model] = new_acc
            else:
                logger.info("Cached model, skipping evaluation...")
                new_acc = model_cache[new_model]

            new_size = new_model.size()
            logger.info("Validation accuracy: " + str(new_acc) + " - Size: " + str(new_size))

            rule_consequent = comparator.compute_fired_rules(best_size, best_acc, new_size, new_acc)
            logger.info("Fired rule consequent - " + str(rule_consequent))

            if rule_consequent == RuleConsequent.NEW_BEST:
                best_model, best_acc, best_size = new_model, new_acc, new_size
                search_space.learn(model=best_model, positive=True)
                search_space.update_previous_state(model=best_model)
                operator_weights[op_idx] += 1
                history.append((explored_models, best_acc, best_size))
                logger.info("Best model updated")

            explored_models += 1
            failed_models = 0

        except Exception as exception:
            unhandled_model(exception, logger, new_model)
            failed_models += 1
            if failed_models > num_iters:
                raise

    logger.info("Evaluating model in test set...")
    reset_model_parameters(best_model.get_blocks())
    best_model, test_acc = evaluator.evaluate_in_test(best_model)
    logger.info("Test set accuracy: " + str(test_acc))

    return best_model, test_acc, history
