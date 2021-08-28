import json
import pandas
import numpy as np
import networkx as nx

from d2d.d2d import orient_edges, ORIENTATION_EPSILON
from runs.features import get_features
from preprocessing.yeast.kegg_example import KEGG_GENERATED_FOLDER

KEGG_NAME = 'sce00010'

PRIOR_WEIGHT_FUNCS = {
    'unweighted': lambda x: 1,
    'abs': abs,
    'exp abs': lambda x: 2 ** abs(x)
}


def run_yeast_kegg_experiment(prior_weight_func_name='abs'):
    network = nx.read_gpickle(KEGG_GENERATED_FOLDER / KEGG_NAME / 'net.gpickle')

    with open(KEGG_GENERATED_FOLDER / KEGG_NAME / 'kegg_annotations.json', 'r') as f:
        true_annotations_list = json.load(f)
        # Represent edges as tuples and not list
        true_annotations = list(map(lambda e: tuple(e), true_annotations_list))

    with open(KEGG_GENERATED_FOLDER / KEGG_NAME / 'experiment.json', 'r') as f:
        experiments = json.load(f)

    experiment = list(experiments.keys())[0]
    feature_cols, reverse_cols = get_features(network, experiments,
                                              prior_weight_func=PRIOR_WEIGHT_FUNCS[prior_weight_func_name],
                                              force=True, save=False)
    scores = pandas.DataFrame(np.column_stack([feature_cols[experiment].values, reverse_cols[experiment].values]),
                              index=feature_cols.index, columns=["(u,v)", "(v,u)"])
    annotated_network, annotated_edges = orient_edges(scores, orientation_epsilon=ORIENTATION_EPSILON)

    true_positive = set(annotated_edges).intersection(true_annotations)
    opposite_edges = list(map(lambda e: (e[1], e[0]), annotated_edges))
    false_annotations = set(opposite_edges).intersection(true_annotations)

    unannotated_num = len(feature_cols.index) - len(annotated_edges)

    print(f"Total edges to annotate: {len(feature_cols.index)}")
    print(f"TP: {len(true_positive)} out of {len(true_annotations)}")
    print(f"false annotations: {len(false_annotations)} out of {len(true_annotations)}")
    print(f"undetermined annotated: {unannotated_num} out of {len(true_annotations)}")


if __name__ == '__main__':
    run_yeast_kegg_experiment(prior_weight_func_name='unweighted')
