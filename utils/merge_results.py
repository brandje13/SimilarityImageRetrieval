def merge_results(cfg,  models, mode):
    results = {}

    for query in cfg['qimlist']:
        joined_set = set()

        for model in models:
            top_k_set = set(model[1][query]['top_k'])

            if len(joined_set) == 0:
                joined_set = top_k_set.copy()
            elif mode == 'union':
                joined_set.update(top_k_set)
            elif mode == 'intersection':
                joined_set.intersection_update(top_k_set)

        results[query] = list(joined_set)

    return results