import os
import pandas
import pathlib

from d2d.propagation import RWR_PROPAGATION, PROPAGATE_ALPHA, PROPAGATE_EPSILON, PROPAGATE_ITERATIONS
from d2d.d2d import generate_feature_columns

FEATURES_FILE_NAME = 'features_cols.pickle'
REVERESE_FILE_NAME = 'reverse_cols.pickle'


def get_features(network, experiments, save=True, force=False, output_directory=pathlib.Path(),
                 alpha=PROPAGATE_ALPHA, epsilon=PROPAGATE_EPSILON,
                 method=RWR_PROPAGATION, num_iterations=PROPAGATE_ITERATIONS,
                 prior_weight_func=abs):
    features_path = output_directory / FEATURES_FILE_NAME
    reverse_path = output_directory / REVERESE_FILE_NAME
    assert os.path.isdir(output_directory), "Given path is not a valid directory"

    if os.path.isdir(output_directory) and not force:
        if os.path.isfile(features_path) and os.path.isfile(reverse_path):
            return pandas.read_pickle(features_path), pandas.read_pickle(reverse_path)

    feature_columns, reverse_columns = \
        generate_feature_columns(network, experiments, alpha=alpha, epsilon=epsilon,
                                 method=method, num_iterations=num_iterations,
                                 prior_weight_func=prior_weight_func)

    if save:
        feature_columns.to_pickle(features_path)
        reverse_columns.to_pickle(reverse_path)

    return feature_columns, reverse_columns
