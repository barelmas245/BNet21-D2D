import json
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from d2d.d2d import get_training_features_and_scores, score_network, orient_edges, ORIENTATION_EPSILON
from runs.yeast.data import read_data
from runs.features import get_features
from runs.yeast.consts import YEAST_RESULTS_DIR


def cross_validation(feature_columns, reverse_columns, directed_interactions):
    directed_feature_columns, directed_reverse_columns, directed_feature_scores, directed_reverse_scores = \
        get_training_features_and_scores(feature_columns, reverse_columns, directed_interactions)

    cv = StratifiedKFold(n_splits=10)
    tps = []
    falses = []
    unannotated = []

    import random
    random.shuffle(directed_interactions)
    directed_interactions_columns = pandas.DataFrame(index=directed_interactions)
    for i, (train, test) in enumerate(cv.split(directed_interactions, np.ones(len(directed_interactions)))):
        print(f"######### Test {i}")
        training_directed_interactions = list(directed_interactions_columns.iloc[train].index)
        test_directed_interactions = list(directed_interactions_columns.iloc[test].index)

        classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.01)
        scores = score_network(directed_feature_columns, directed_reverse_columns,
                               training_directed_interactions, classifier)
        annotated_network, annotated_edges = orient_edges(scores, orientation_epsilon=ORIENTATION_EPSILON)

        # assert set(annotated_edges).intersection(training_directed_interactions) == set(training_directed_interactions)
        new_annotated_edges = set(annotated_edges).difference(training_directed_interactions)
        true_positive = new_annotated_edges.intersection(test_directed_interactions)
        opposite_edges = list(map(lambda e: (e[1], e[0]), new_annotated_edges))
        false_annotations = set(opposite_edges).intersection(test_directed_interactions)
        unannotated_num = len(test_directed_interactions) - len(true_positive) - len(false_annotations)

        print(f"Total edges to annotate: {len(test_directed_interactions)}")
        print(f"TP: {len(true_positive)} out of {len(test_directed_interactions)}")
        print(f"false annotations: {len(false_annotations)} out of {len(test_directed_interactions)}")
        print(f"undetermined annotated: {unannotated_num} out of {len(test_directed_interactions)}")

        tps.append(len(true_positive) / len(test_directed_interactions))
        falses.append(len(false_annotations) / len(test_directed_interactions))
        unannotated.append(unannotated_num / len(test_directed_interactions))

    print("################ Results")
    print(f"mean true positives: {np.mean(tps)}")
    print(f"mean false annotations: {np.mean(falses)}")
    print(f"mean unannotated: {np.mean(unannotated)}")


if __name__ == '__main__':
    net_type = 'anat'
    undirected = True
    network, true_annotations, experiments, edges_to_direct = read_data(net_type=net_type, undirected=undirected)

    print(f"Network {net_type}, undirected={undirected}: nodes: {len(network.nodes)}, edges: {len(network.edges)}")
    print(f"Total edges to direct: {len(edges_to_direct) if edges_to_direct else 'Unknown'}")
    print(f"Test annotations: {len(true_annotations)}")

    feature_cols, reverse_cols = get_features(network, experiments, edges_to_direct=edges_to_direct,
                                              prior_weight_func=lambda x: abs(x), output_directory=YEAST_RESULTS_DIR,
                                              force=True, save=True)
    cross_validation(feature_cols, reverse_cols, true_annotations)
