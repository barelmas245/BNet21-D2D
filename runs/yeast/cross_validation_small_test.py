import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from d2d.d2d import get_training_features_and_scores, score_network, orient_edges, ORIENTATION_EPSILON
from runs.features import get_features


def cross_validation(feature_columns, reverse_columns, directed_interactions):
    directed_feature_columns, directed_reverse_columns, directed_feature_scores, directed_reverse_scores = \
        get_training_features_and_scores(feature_columns, reverse_columns, directed_interactions)

    cv = StratifiedKFold(n_splits=2)
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

        classifier = LogisticRegression(solver="liblinear", penalty="l1", C=2)
        # classifier = LogisticRegressionCV(solver="liblinear", penalty="l1", cv=3)
        scores = score_network(directed_feature_columns, directed_reverse_columns,
                               training_directed_interactions, classifier)
        annotated_network, annotated_edges = orient_edges(scores, orientation_epsilon=ORIENTATION_EPSILON)

        # assert set(annotated_edges).intersection(training_directed_interactions) == set(training_directed_interactions)
        new_annotated_edges = set(annotated_edges).difference(training_directed_interactions)
        true_positive = new_annotated_edges.intersection(test_directed_interactions)
        false_annotations = new_annotated_edges.difference(test_directed_interactions)
        unannotated_num = len(directed_interactions) - len(annotated_edges)
        print(f"TP: {len(true_positive)} out of {len(test_directed_interactions)}")
        print(f"false annotations: {len(false_annotations)} out of {len(test_directed_interactions)}")
        print(f"Not annotated: {unannotated_num} out of {len(test_directed_interactions)}")
        tps.append(len(true_positive) / len(test_directed_interactions))
        falses.append(len(false_annotations) / len(test_directed_interactions))
        unannotated.append(unannotated_num / len(test_directed_interactions))

    print("################ Results")
    print(f"mean true positives: {np.mean(tps)}")
    print(f"mean false annotations: {np.mean(falses)}")
    print(f"mean unannotated: {np.mean(unannotated)}")


if __name__ == '__main__':
    true_annotations = [
        ("1", "4"), ("2", "4"), ("4", "5"), ("3", "4"), ("5", "7"), ("6", "7"), ("7", "8"), ("7", "9"),
        ("4", "10"), ("9", "11"), ("9", "12"), ("12", "13"), ("11", "14"), ("11", "15"), ("15", "16")
    ]

    import networkx as nx
    network = nx.Graph()
    network.add_edges_from(true_annotations)

    sources = ["1", "2", "3", "6"]
    targets = ["8", "10", "13", "14", "16"]

    experiments = {
        "1": {
            "8": -1.5,
            "10": -1.2,
            "13": -0.8,
            "14": -0.75,
            "16": -0.7
        },
        "2": {
            "8": -1.6,
            "10": -1.1,
            "13": -0.65,
            "14": -0.8,
            "16": -0.75
        },
        "3": {
            "8": -1.5,
            "13": -0.9,
            "14": -0.9,
            "16": -0.8
        },
        "6": {
            "8": -1,
            "13": -0.5,
            "14": -0.7,
            "16": -0.8
        }
    }

    feature_cols, reverse_cols = get_features(network, experiments, save=False, force=True)
    cross_validation(feature_cols, reverse_cols, true_annotations)
