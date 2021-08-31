import copy
import pandas
import numpy as np
import networkx as nx

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
