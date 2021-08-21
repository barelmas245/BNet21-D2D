import networkx as nx
from sklearn.linear_model import LogisticRegression

from d2d.d2d import read_data, score_network, orient_network
from runs.features import get_features
from runs.consts import FEATURE_COLS_PATH, REVERSE_COLS_PATH, ANNOTATED_NETWORK_PATH


if __name__ == '__main__':
    network, true_annotations, experiments = read_data()
    feature_columns, reverse_columns = get_features(network, experiments,
                                                    FEATURE_COLS_PATH, REVERSE_COLS_PATH)
    classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001)
    scores = score_network(feature_columns, reverse_columns, true_annotations, classifier)
    annotated_network, annotated_edges = orient_network(network, scores)
    nx.write_gpickle(annotated_network, ANNOTATED_NETWORK_PATH)
