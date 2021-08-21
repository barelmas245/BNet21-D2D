import os
import pandas

from d2d.propagation import RWR_PROPAGATION, PROPAGATE_ALPHA, PROPAGATE_EPSILON, PROPAGATE_ITERATIONS
from d2d.d2d import generate_feature_columns


def get_features(network, experiments, features_path, reverse_features):
    if os.path.isfile(features_path) and os.path.isfile(reverse_features):
        return pandas.read_pickle(features_path), pandas.read_pickle(reverse_features)
    else:
        feature_columns, reverse_columns = \
            generate_feature_columns(network, experiments, alpha=PROPAGATE_ALPHA, epsilon=PROPAGATE_EPSILON,
                                     method=RWR_PROPAGATION, num_iterations=PROPAGATE_ITERATIONS,
                                     prior_weight_func=abs)
        feature_columns.to_pickle(features_path)
        reverse_columns.to_pickle(reverse_features)
        return feature_columns, reverse_columns
