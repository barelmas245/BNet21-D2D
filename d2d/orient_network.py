import os
import pandas
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve, auc
import matplotlib.pyplot as plt

from preprocessing.biogrid.read_biogrid import get_biogrid_network
from preprocessing.breikreutz.experiments import get_gene_expressions_data
from preprocessing.breikreutz.annotations import get_true_annotations

from consts import FEATURE_COLS_PATH, REVERSE_COLS_PATH, CROSS_VALIDATION_ROC_PATH, ANNOTATED_NETWORK_PATH,\
    ORIENTATION_EPSILON
from propagate import generate_similarity_matrix, propagate, \
    RWR_PROPAGATION, PROPAGATE_ALPHA, PROPAGATE_EPSILON, PROPAGATE_ITERATIONS, PROPAGATE_SMOOTH


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

    feature_columns = pandas.DataFrame(np.column_stack([generate_column(experiment) for experiment in experiments]),
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


def orient_network(network, scores):
    edges = network.edges
    assert set(edges) == set(scores.index)

    potential_oriented_edges = scores[scores['(u,v)'] >= scores['(v,u)']].index
    potential_inverted_edges = scores[scores['(u,v)'] < scores['(v,u)']].index

    scores_ratio = pandas.concat([scores['(u,v)'] / scores['(v,u)'], scores['(v,u)'] / scores['(u,v)']]).max(level=0)

    edges_to_annotate = scores_ratio[scores_ratio > 1 + ORIENTATION_EPSILON].index

    oriented_edges = list(potential_oriented_edges.intersection(edges_to_annotate))
    inverted_edges = list(map(lambda e: (e[1], e[0]), potential_inverted_edges.intersection(edges_to_annotate)))

    annotated_edges = oriented_edges
    annotated_edges.extend(inverted_edges)

    directed_network = nx.DiGraph()
    directed_network.add_edges_from(annotated_edges)

    return directed_network, annotated_edges


def cross_validation(feature_columns, reverse_columns, directed_interactions, classifier):
    opposite_directed_interactions = list(map(lambda e: (e[1], e[0]), directed_interactions))

    true_in_feature = set(directed_interactions).intersection(feature_columns.index)
    false_in_feature = set(opposite_directed_interactions).intersection(feature_columns.index)

    training_columns = pandas.concat([feature_columns.loc[true_in_feature, :], feature_columns.loc[false_in_feature, :],
                                      reverse_columns.loc[false_in_feature, :],
                                      reverse_columns.loc[true_in_feature, :]])

    true_feature_labels = np.ones(len(true_in_feature))
    false_feature_labels = np.zeros(len(false_in_feature))
    true_reverse_labels = np.ones(len(false_in_feature))
    false_reverse_labels = np.zeros(len(true_in_feature))

    feature_labels = np.append(true_feature_labels, false_feature_labels)
    reverse_labels = np.append(true_reverse_labels, false_reverse_labels)
    training_scores = np.append(feature_labels, reverse_labels)

    cv = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(training_columns, training_scores)):
        classifier.fit(training_columns.iloc[train], training_scores[train])
        viz = plot_roc_curve(classifier, training_columns.iloc[test], training_scores[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC curve")
    ax.legend(loc="lower right")
    plt.savefig(CROSS_VALIDATION_ROC_PATH)


if __name__ == '__main__':
    network, true_annotations, experiments = read_data()
    true_annotations = list(map(lambda e: tuple(e), true_annotations))

    # TODO: assert that all true directed interacions are inside the network

    if os.path.isfile(FEATURE_COLS_PATH) and os.path.isfile(REVERSE_COLS_PATH):
        feature_columns = pandas.read_pickle(FEATURE_COLS_PATH)
        reverse_columns = pandas.read_pickle(REVERSE_COLS_PATH)
    else:
        feature_columns, reverse_columns = \
            generate_feature_columns(network, experiments, alpha=PROPAGATE_ALPHA, epsilon=PROPAGATE_EPSILON,
                                     method=RWR_PROPAGATION, num_iterations=PROPAGATE_ITERATIONS,
                                     prior_weight_func=abs)
        feature_columns.to_pickle(FEATURE_COLS_PATH)
        reverse_columns.to_pickle(REVERSE_COLS_PATH)

    classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001)

    # cross_validation(feature_columns, reverse_columns, true_annotations, classifier)

    scores = score_network(feature_columns, reverse_columns, true_annotations, classifier)
    annotated_network, annotated_edges = orient_network(network, scores)
    nx.write_gpickle(annotated_network, ANNOTATED_NETWORK_PATH)
