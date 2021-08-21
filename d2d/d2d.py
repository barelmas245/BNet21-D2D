import pandas
import numpy as np
import networkx as nx

from preprocessing.biogrid.read_biogrid import get_biogrid_network
from preprocessing.breikreutz.experiments import get_gene_expressions_data
from preprocessing.breikreutz.annotations import get_true_annotations

from d2d.propagation import generate_similarity_matrix, propagate, \
    RWR_PROPAGATION, PROPAGATE_ALPHA, PROPAGATE_EPSILON, PROPAGATE_ITERATIONS, PROPAGATE_SMOOTH

ORIENTATION_EPSILON = 0.01


def read_data():
    network = get_biogrid_network()
    gene_expressions = get_gene_expressions_data()
    directed_interactions = get_true_annotations()
    return network, directed_interactions, gene_expressions


def generate_feature_columns(network, experiments_dict,
                             alpha=PROPAGATE_ALPHA, epsilon=PROPAGATE_EPSILON, method=RWR_PROPAGATION,
                             num_iterations=PROPAGATE_ITERATIONS, smooth=PROPAGATE_SMOOTH,
                             prior_weight_func=abs):
    gene_to_idx = dict(map(lambda e: (e[1], e[0]), enumerate(network.nodes)))
    matrix = generate_similarity_matrix(network, alpha=alpha, method=method)

    # All edges endpoints in the graph
    u_nodes = list(map(lambda e: e[0], network.edges))
    v_nodes = list(map(lambda e: e[1], network.edges))
    u_indexes = list(map(lambda node: gene_to_idx[node], u_nodes))
    v_indexes = list(map(lambda node: gene_to_idx[node], v_nodes))

    def generate_column(experiment):
        # We always have only single source and multiple targets
        sources_dict = {experiment: 1}
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

    feature_columns = pandas.DataFrame(np.column_stack([generate_column(experiment) for experiment in experiments_dict]),
                                       index=list(zip(u_nodes, v_nodes)), columns=list(experiments_dict.keys()))
    reverse_columns = (1 / feature_columns).replace(np.inf, np.nan).fillna(0)
    return feature_columns, reverse_columns


def score_network(feature_columns, reverse_columns, directed_interactions, classifier):
    opposite_directed_interactions = list(map(lambda e: (e[1], e[0]), directed_interactions))

    true_in_feature = set(directed_interactions).intersection(feature_columns.index)
    false_in_feature = set(opposite_directed_interactions).intersection(feature_columns.index)

    training_columns = pandas.concat([feature_columns.loc[true_in_feature, :], feature_columns.loc[false_in_feature, :],
                                      reverse_columns.loc[false_in_feature, :], reverse_columns.loc[true_in_feature, :]])

    true_feature_labels = np.ones(len(true_in_feature))
    false_feature_labels = np.zeros(len(false_in_feature))
    true_reverse_labels = np.ones(len(false_in_feature))
    false_reverse_labels = np.zeros(len(true_in_feature))

    feature_labels = np.append(true_feature_labels, false_feature_labels)
    reverse_labels = np.append(true_reverse_labels, false_reverse_labels)
    training_scores = np.append(feature_labels, reverse_labels)

    fitted_classifier = classifier.fit(training_columns, training_scores)

    unclassified_feature_scores = fitted_classifier.predict_proba(feature_columns.drop(training_columns.index))[:, 1]
    unclassified_reverse_scores = fitted_classifier.predict_proba(reverse_columns.drop(training_columns.index))[:, 1]

    feature_scores = np.concatenate((true_feature_labels, false_feature_labels, unclassified_feature_scores))
    reverse_scores = np.concatenate((true_reverse_labels, false_reverse_labels, unclassified_reverse_scores))

    feature_index = np.concatenate((list(true_in_feature), list(false_in_feature),
                                       list(feature_columns.index.difference(training_columns.index))))
    feature_data_frame = pandas.DataFrame(feature_scores, index=feature_index, columns=["score"]).sort_index()

    reverse_index = np.concatenate((list(false_in_feature), list(true_in_feature),
                                       list(reverse_columns.index.difference(training_columns.index))))
    reverse_data_frame = pandas.DataFrame(reverse_scores, index=reverse_index, columns=["score"]).sort_index()

    assert feature_data_frame.index.equals(reverse_data_frame.index)

    return pandas.DataFrame(np.column_stack([feature_data_frame["score"].values, reverse_data_frame["score"].values]),
                            index=feature_data_frame.index,
                            columns=["(u,v)", "(v,u)"])


def orient_network(network, scores, orientation_epsilon=ORIENTATION_EPSILON):
    edges = network.edges
    assert set(edges) == set(scores.index)

    potential_oriented_edges = scores[scores['(u,v)'] >= scores['(v,u)']].index
    potential_inverted_edges = scores[scores['(u,v)'] < scores['(v,u)']].index

    scores_ratio = pandas.concat([scores['(u,v)'] / scores['(v,u)'], scores['(v,u)'] / scores['(u,v)']]).max(level=0)

    edges_to_annotate = scores_ratio[scores_ratio > 1 + orientation_epsilon].index

    oriented_edges = list(potential_oriented_edges.intersection(edges_to_annotate))
    inverted_edges = list(map(lambda e: (e[1], e[0]), potential_inverted_edges.intersection(edges_to_annotate)))

    annotated_edges = oriented_edges
    annotated_edges.extend(inverted_edges)

    directed_network = nx.DiGraph()
    directed_network.add_edges_from(annotated_edges)

    return directed_network, annotated_edges
