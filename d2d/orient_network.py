import math
import numpy
import pandas
import scipy
import networkx as nx
from sklearn.linear_model import LogisticRegression

from network.biogrid.read_biogrid import get_biogrid_network
from breikreutz.experiments import get_gene_expressions_data
from breikreutz.annotations import get_true_annotations

PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100

ORIENTATION_EPSILON = 0.01


def read_data():
    network = get_biogrid_network()
    gene_expressions = get_gene_expressions_data()

    directed_interactions = get_true_annotations()

    return network, directed_interactions, gene_expressions


def generate_similarity_matrix(graph):
    matrix = nx.normalized_laplacian_matrix(graph)

    # matrix = adjacency_matrix(graph)
    # norm_matrix = diags(1 / numpy.sqrt(matrix.sum(0).A1))
    # matrix = norm_matrix * matrix * norm_matrix

    return PROPAGATE_ALPHA * matrix


def propagate(seeds_dict, matrix, gene_to_index):
    num_genes = matrix.shape[0]
    curr_scores = numpy.zeros(num_genes)

    # Set the scores seeds
    for gene in seeds_dict:
        if gene in gene_to_index:
            # We do not differ between over/under expression
            curr_scores[gene_to_index[gene]] = abs(seeds_dict[gene])
        else:
            print(f"Not found gene {gene} in network!")
            raise RuntimeError()
    # Normalize the prior seeds
    curr_scores = curr_scores / sum(curr_scores)

    prior_vec = (1 - PROPAGATE_ALPHA) * curr_scores

    for _ in range(PROPAGATE_ITERATIONS):
        new_scores = curr_scores.copy()
        curr_scores = matrix.dot(new_scores) + prior_vec

        if math.sqrt(scipy.linalg.norm(new_scores - curr_scores)) < PROPAGATE_EPSILON:
            break

    return curr_scores


def generate_propagation_data(network):
    matrix = generate_similarity_matrix(network)
    gene_to_idx = dict(map(lambda e: (e[1], e[0]), enumerate(network.nodes)))
    return gene_to_idx, matrix


def generate_feature_columns(network, experiments_dict, true_annotations):
    gene_to_idx, matrix = generate_propagation_data(network)

    # All edges in the graph
    u_nodes = list(map(lambda e: e[0], network.edges))
    v_nodes = list(map(lambda e: e[1], network.edges))
    u_indexes = list(map(lambda node: gene_to_idx[node], u_nodes))
    v_indexes = list(map(lambda node: gene_to_idx[node], v_nodes))

    # Only the edges we have knowledge of
    training_u_indexes = list(map(lambda e: gene_to_idx[e[0]], true_annotations))
    training_v_indexes = list(map(lambda e: gene_to_idx[e[1]], true_annotations))

    def generate_column(experiment):
        # We always have only single source and multiple targets
        sources_dict = {experiment: 1}
        terminals_dict = experiments_dict[experiment]

        print(f"Running propagation of knockout {experiment}")
        source_scores = propagate(sources_dict, matrix, gene_to_idx)
        terminal_scores = propagate(terminals_dict, matrix, gene_to_idx)

        cause_u_scores = source_scores[u_indexes]
        cause_v_scores = source_scores[v_indexes]
        effect_u_scores = terminal_scores[u_indexes]
        effect_v_scores = terminal_scores[v_indexes]

        return numpy.nan_to_num((cause_u_scores * effect_v_scores) / (effect_u_scores * cause_v_scores))

    feature_columns = pandas.DataFrame(numpy.column_stack([generate_column(experiment) for experiment in experiments]),
                                       index=list(zip(u_nodes, v_nodes)), columns=list(experiments_dict.keys()))
    reverse_columns = (1 / feature_columns).replace(numpy.inf, numpy.nan).fillna(0)
    return feature_columns, reverse_columns


def score_network(feature_columns, reverse_columns, directed_interactions):
    opposite_directed_interactions = list(map(lambda e: (e[1], e[0]), directed_interactions))

    true_in_feature = set(directed_interactions).intersection(feature_columns.index)
    true_in_reverse = set(opposite_directed_interactions).intersection(reverse_columns.index)
    false_in_feature = set(opposite_directed_interactions).intersection(feature_columns.index)
    false_in_reverse = set(directed_interactions).intersection(reverse_columns.index)

    training_columns = pandas.concat([feature_columns.loc[true_in_feature, :], feature_columns.loc[false_in_feature, :],
                                      reverse_columns.loc[true_in_reverse, :], reverse_columns.loc[false_in_reverse, :]])

    true_feature_labels = numpy.ones(len(true_in_feature))
    false_feature_labels = numpy.zeros(len(false_in_feature))
    true_reverse_labels = numpy.ones(len(true_in_feature))
    false_reverse_labels = numpy.zeros(len(false_in_feature))

    feature_labels = numpy.append(true_feature_labels, false_feature_labels)
    reverse_labels = numpy.append(true_reverse_labels, false_reverse_labels)
    training_scores = numpy.append(feature_labels, reverse_labels)

    classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001).fit(training_columns, training_scores)

    unclassified_feature_scores = classifier.predict_proba(feature_columns.drop(true_in_feature.union(false_in_feature)))[:, 1]
    unclassified_reverse_scores = classifier.predict_proba(reverse_columns.drop(true_in_reverse.union(false_in_reverse)))[:, 1]

    feature_scores = numpy.concatenate((true_feature_labels, false_feature_labels, unclassified_feature_scores))
    reverse_scores = numpy.concatenate((true_reverse_labels, false_reverse_labels, unclassified_reverse_scores))

    feature_index = numpy.concatenate((list(true_in_feature), list(false_in_feature),
                                       list(set(feature_columns.index).difference(
                                           true_in_feature.union(false_in_feature)))))
    feature_data_frame = pandas.DataFrame(feature_scores, index=feature_index, columns=["score"]).sort_index()

    reverse_index = numpy.concatenate((list(true_in_reverse), list(false_in_reverse),
                                       list(set(reverse_columns.index).difference(
                                           true_in_reverse.union(false_in_reverse)))))
    reverse_data_frame = pandas.DataFrame(reverse_scores, index=reverse_index, columns=["score"]).sort_index()

    assert feature_data_frame.index.equals(reverse_data_frame.index)

    return pandas.DataFrame(numpy.column_stack([feature_data_frame["score"].values, reverse_data_frame["score"].values]),
                            index=feature_data_frame.index,
                            columns=["(u,v)", "(v,u)"])


def orient_network(network, scores):

    inverted = network[scores[2] < scores[3]]
    network[3] = scores[2]/scores[3]
    network.loc[inverted.index, 3] = scores[3]/scores[2]

    oriented = network[network[3] > 1 + ORIENTATION_EPSILON]
    network[4] = 0
    network.loc[oriented.index, 4] = 1

    swapped = inverted.index & oriented.index
    network.loc[swapped, [0,1]] = network.loc[swapped, [1,0]].values

    return network.set_index([0,1])


if __name__ == '__main__':
    network, true_annotations, experiments = read_data()
    true_annotations = list(map(lambda e: tuple(e), true_annotations))

    # TODO: check that all true directed interacions are inside the network
    # merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])

    feature_columns, reverse_columns = generate_feature_columns(network, experiments, true_annotations)
    scores = score_network(feature_columns, reverse_columns, true_annotations)
    oriented_network = orient_network(network, scores)
    pass