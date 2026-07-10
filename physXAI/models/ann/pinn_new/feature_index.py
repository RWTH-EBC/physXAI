def _resolve_feature_indices(features, columns):
    indices = []
    for feature in features:
        if isinstance(feature, str):
            indices.append(list(columns).index(feature))
        else:
            indices.append(feature)
    return indices
