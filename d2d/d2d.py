import copy
import pandas
import numpy as np
import networkx as nx

from sklearn.metrics import precision_recall_curve
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from d2d.propagation import generate_similarity_matrix, propagate, \
    RWR_PROPAGATION, PROPAGATE_ALPHA, PROPAGATE_EPSILON, PROPAGATE_ITERATIONS, PROPAGATE_SMOOTH

ORIENTATION_EPSILON = 0.01


def generate_feature_columns(network, experiments_dict, edges_to_direct=None,
                             alpha=PROPAGATE_ALPHA, epsilon=PROPAGATE_EPSILON, method=RWR_PROPAGATION,
                             num_iterations=PROPAGATE_ITERATIONS, smooth=PROPAGATE_SMOOTH,
                             prior_weight_func=abs):
    gene_to_idx = dict(map(lambda e: (e[1], e[0]), enumerate(network.nodes)))
    matrix = generate_similarity_matrix(network, alpha=alpha, method=method)

    # All edges endpoints in the graph
    if edges_to_direct:
        distinct_edges = edges_to_direct
    else:
        distinct_edges = network.edges
        if isinstance(network, nx.DiGraph):
            distinct_edges = set(distinct_edges)
            distinct_edges = list(distinct_edges.intersection(
                list(map(lambda e: (e[1], e[0]), distinct_edges))))
            distinct_edges = list(filter(lambda e: (e[1], e[0]) not in distinct_edges, distinct_edges))

    u_nodes = list(map(lambda e: e[0], distinct_edges))
    v_nodes = list(map(lambda e: e[1], distinct_edges))
    u_indexes = list(map(lambda node: gene_to_idx[node], u_nodes))
    v_indexes = list(map(lambda node: gene_to_idx[node], v_nodes))

    def generate_column(experiment):
        all_sources = experiment.split(',')
        sources_dict = dict(map(lambda source: (source, 1), all_sources))
        terminals_dict = experiments_dict[experiment]

        print(f"Running propagation of knockout {experiment}")
        source_scores = propagate(sources_dict, matrix, gene_to_idx, alpha=alpha, epsilon=epsilon,
                                  method=method, num_iterations=num_iterations, smooth=smooth,
                                  prior_weight_func=prior_weight_func)
        terminal_scores = propagate(terminals_dict, matrix, gene_to_idx, alpha=alpha, epsilon=epsilon,
                                    method=method, num_iterations=num_iterations, smooth=smooth,
                                    prior_weight_func=prior_weight_func)

        cause_u_scores = source_scores[u_indexes]
        cause_v_scores = source_scores[v_indexes]
        effect_u_scores = terminal_scores[u_indexes]
        effect_v_scores = terminal_scores[v_indexes]
        return np.nan_to_num((cause_u_scores * effect_v_scores) / (effect_u_scores * cause_v_scores))

    feature_columns = pandas.DataFrame(np.column_stack([
        generate_column(experiment) for experiment in experiments_dict]),
        index=list(zip(u_nodes, v_nodes)), columns=list(experiments_dict.keys()))
    reverse_columns = (1 / feature_columns).replace(np.inf, np.nan).fillna(0)
    reverse_columns = reverse_columns.set_axis(list(zip(v_nodes, u_nodes)), axis=0)
    return feature_columns, reverse_columns


def get_training_features_and_scores(feature_columns, reverse_columns, directed_interactions):
    opposite_directed_interactions = list(map(lambda e: (e[1], e[0]), directed_interactions))

    true_in_feature = set(directed_interactions).intersection(feature_columns.index)
    false_in_feature = set(opposite_directed_interactions).intersection(feature_columns.index)

    true_in_reverse = set(directed_interactions).intersection(reverse_columns.index)
    false_in_reverse = set(opposite_directed_interactions).intersection(reverse_columns.index)

    training_feature_columns = pandas.concat([feature_columns.loc[true_in_feature, :],
                                              feature_columns.loc[false_in_feature, :]])
    training_reverse_columns = pandas.concat([reverse_columns.loc[true_in_reverse, :],
                                              reverse_columns.loc[false_in_reverse, :]])

    true_feature_labels = np.ones(len(true_in_feature))
    false_feature_labels = np.zeros(len(false_in_feature))
    true_reverse_labels = np.ones(len(true_in_reverse))
    false_reverse_labels = np.zeros(len(false_in_reverse))

    training_feature_scores = np.append(true_feature_labels, false_feature_labels)
    training_reverse_scores = np.append(true_reverse_labels, false_reverse_labels)

    return training_feature_columns, training_reverse_columns, training_feature_scores, training_reverse_scores


def score_network(feature_columns, reverse_columns, directed_interactions, classifier):
    training_feature_columns, training_reverse_columns, training_feature_scores, training_reverse_scores = \
        get_training_features_and_scores(feature_columns, reverse_columns, directed_interactions)

    training_columns = pandas.concat([training_feature_columns, training_reverse_columns])
    training_scores = np.append(training_feature_scores, training_reverse_scores)

    fitted_classifier = classifier.fit(training_columns, training_scores)

    unclassified_feature_scores = fitted_classifier.predict_proba(feature_columns.drop(training_feature_columns.index))[:, 1]
    unclassified_reverse_scores = fitted_classifier.predict_proba(reverse_columns.drop(training_reverse_columns.index))[:, 1]

    feature_scores = np.concatenate((training_feature_scores, unclassified_feature_scores))
    reverse_scores = np.concatenate((training_reverse_scores, unclassified_reverse_scores))

    feature_index = copy.copy(list(training_feature_columns.index))
    feature_index.extend(list(feature_columns.index.difference(training_columns.index)))
    feature_data_frame = pandas.DataFrame(feature_scores, index=feature_index, columns=["score"]).sort_index()

    reverse_index = copy.copy(list(training_reverse_columns.index))
    reverse_index.extend(list(reverse_columns.index.difference(training_columns.index)))
    reverse_data_frame = pandas.DataFrame(reverse_scores, index=reverse_index, columns=["score"])
    opposite_edges = list(map(lambda e: (e[1], e[0]), list(feature_data_frame.index)))
    reverse_data_frame = reverse_data_frame.reindex(opposite_edges)

    return pandas.DataFrame(np.column_stack([feature_data_frame["score"].values, reverse_data_frame["score"].values]),
                            index=feature_data_frame.index,
                            columns=["(u,v)", "(v,u)"])


def orient_edges(scores, orientation_epsilon=ORIENTATION_EPSILON):
    potential_oriented_edges = scores[scores['(u,v)'] >= scores['(v,u)']].index
    potential_inverted_edges = scores[scores['(u,v)'] < scores['(v,u)']].index

    scores_ratio = pandas.concat([scores['(u,v)'] / scores['(v,u)'], scores['(v,u)'] / scores['(u,v)']]).max(level=0)

    edges_to_annotate = scores_ratio[scores_ratio > 1 + orientation_epsilon].index

    oriented_edges = list(potential_oriented_edges.intersection(edges_to_annotate))
    inverted_edges = list(map(lambda e: (e[1], e[0]), potential_inverted_edges.intersection(edges_to_annotate)))
    assert set(oriented_edges).intersection(inverted_edges) == set()
    annotated_edges = oriented_edges
    annotated_edges.extend(inverted_edges)

    directed_network = nx.DiGraph()
    directed_network.add_edges_from(annotated_edges)

    return directed_network, annotated_edges


# TODO: precision & recall
# def get_recall_precision(feature_columns, reverse_columns, directed_interactions):
#     directed_feature_columns, directed_reverse_columns, directed_feature_scores, directed_reverse_scores = \
#         get_training_features_and_scores(feature_columns, reverse_columns, directed_interactions)
#
#     cv = StratifiedKFold(n_splits=12)
#
#     all_proba = []
#     all_ytest = []
#     all_names = []
#
#     sum_coef_zero = 0
#
#     # Choose folds so tonot mix the true instances with the same interactions' false
#     folds = []
#     import random
#     random.shuffle(directed_interactions)
#     directed_interactions_columns = pandas.DataFrame(index=directed_interactions)
#     for i, (train_index, test_index) in enumerate(cv.split(directed_interactions, np.ones(len(directed_interactions)))):
#         training_directed_interactions = directed_interactions_columns.iloc[train_index].index
#         test_directed_interactions = directed_interactions_columns.iloc[test_index].index
#         # # Choose the corresponding false (this is done to prevent information leak caused by having one direction chosen
#         # # once in the test set and the opposite direction chosen for the training)
#         # EdgeTrue = (pred_ID2.iloc[train_index].astype(str) + "." + pred_ID1.iloc[train_index].astype(str))
#         # IdxFalse = pred_name[pred_name.isin(EdgeTrue)].index
#         # IdxFullTrain = np.append(IdxFalse.values, train_index)
#         #
#         # # The rest are the training
#         # IdxFullTest = pred_name[~pred_name.index.isin(IdxFullTrain)].index
#         folds.append((training_directed_interactions, test_directed_interactions))
#
#     for train_index, test_index in folds:
#         xtrain, xtest = directed_feature_columns.loc[train_index], feature_columns.loc[test_index]
#         # ytrain, ytest = pred_true.loc[train_index], pred_true.loc[test_index]
#         # # name_test = pred_name.loc[test_index]
#         # xtrain = xtrain.rank(axis='columns')
#         # xtest = xtest.rank(axis='columns')
#
#         training_feature_columns, training_reverse_columns, training_feature_scores, training_reverse_scores = \
#             get_training_features_and_scores(directed_feature_columns, directed_reverse_columns, directed_interactions)
#
#         training_columns = pandas.concat([training_feature_columns, training_reverse_columns])
#         training_scores = np.append(training_feature_scores, training_reverse_scores)
#
#         clf = LogisticRegression(solver="liblinear", penalty="l1", C=0.007)
#         score_network(directed_interactions_columns, reverse_columns, directed_interactions, classifier):
#         test_prob = clf.fit(xtrain, ytrain).predict_proba(xtest)
#
#         all_proba.extend(test_prob)
#         all_ytest.extend(ytest)
#         # all_names.extend(name_test)
#
#     # roc_curve
#     precision, recall, thresholds = precision_recall_curve(np.array(all_ytest), np.array(all_proba)[:, 1], pos_label=2)
#     mean_auc = auc(recall, precision)
#     rows, columns = feature_columns.shape
#
#     oriented_network = pandas.concat([pandas.DataFrame({'edge': all_names}), pandas.DataFrame({'edge': all_ytest}),
#                                       pandas.DataFrame({'pval': np.array(all_proba)[:, 1]})], axis=1)
#     oriented_network.columns = ['edge', 'true', 'pval']
#
#     return recall, precision, mean_auc, oriented_network, np.array(all_proba)[:, 1]
