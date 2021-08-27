import json
import pandas
import numpy as np
import networkx as nx

from d2d.d2d import orient_edges, ORIENTATION_EPSILON
from runs.features import get_features
from runs.human.consts import HUMAN_RESULTS_DIR

from preprocessing.human.consts import KEGG_NAME, GENERATED_HUMAN_NET_PATH, GENERATED_KEGG_ANNOTATIONS_PATH, GENERATED_EXPERIMENT_PATH, \
    GENERATED_KEGG_TRUE_DIRECTION_ANNOTATIONS_PATH, GENERATED_KEGG_FALSE_DIRECTION_ANNOTATIONS_PATH


PRIOR_WEIGHT_FUNCS = {
    'unweighted': lambda x: 1,
    'abs': abs,
    'exp abs': lambda x: 2 ** abs(x)
}


def run_human_kegg_experiment(prior_weight_func_name='abs'):
    network = nx.read_gpickle(GENERATED_HUMAN_NET_PATH)

    with open(GENERATED_KEGG_ANNOTATIONS_PATH, 'r') as f:
        true_annotations_list = json.load(f)
        # Represent edges as tuples and not list
        true_annotations = list(map(lambda e: tuple(e), true_annotations_list))

    with open(GENERATED_EXPERIMENT_PATH, 'r') as f:
        experiments = json.load(f)

    experiment = list(experiments.keys())[0]
    feature_cols, reverse_cols = get_features(network, experiments,
                                              prior_weight_func=PRIOR_WEIGHT_FUNCS[prior_weight_func_name],
                                              output_directory=HUMAN_RESULTS_DIR / KEGG_NAME / prior_weight_func_name,
                                              force=True, save=True)
    scores = pandas.DataFrame(np.column_stack([feature_cols[experiment].values, reverse_cols[experiment].values]),
                              index=feature_cols.index, columns=["(u,v)", "(v,u)"])
    annotated_network, annotated_edges = orient_edges(scores, orientation_epsilon=ORIENTATION_EPSILON)

    true_positive = set(annotated_edges).intersection(true_annotations)
    opposite_edges = list(map(lambda e: (e[1], e[0]), annotated_edges))
    false_annotations = set(opposite_edges).intersection(true_annotations)

    new_annotated_num = len(set(annotated_edges).difference(true_annotations))
    unannotated_num = len(feature_cols.index) - len(annotated_edges)

    print(f"Total edges to annotate: {len(feature_cols.index)}")
    print(f"TP: {len(true_positive)} out of {len(true_annotations)}")
    print(f"false annotations: {len(false_annotations)} out of {len(true_annotations)}")
    print(f"undetermined annotated: {unannotated_num} out of {len(true_annotations)}")
    print(f"New annotations: {new_annotated_num} out of {len(feature_cols.index) - len(true_annotations)}")

    with open(GENERATED_KEGG_TRUE_DIRECTION_ANNOTATIONS_PATH, 'r') as f:
        l = json.load(f)
        # Represent edges as tuples and not list
        true_direction_edges = list(map(lambda e: tuple(e), l))
    with open(GENERATED_KEGG_FALSE_DIRECTION_ANNOTATIONS_PATH, 'r') as f:
        l = json.load(f)
        # Represent edges as tuples and not list
        false_direction_edges = list(map(lambda e: tuple(e), l))

    def get_type(e):
        if e in true_direction_edges:
            return 'TRUE_KEGG'
        elif e in false_direction_edges:
            return 'FALSE_KEGG'
        else:
            return 'UNDIRECTED_KEGG'

    scores['type'] = list(map(get_type, scores.index))

    import seaborn as sns
    sns.set_style("whitegrid", {'axes.grid': False})
    import matplotlib

    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    scores['log(D2D(u,v))'] = np.log10(scores['(u,v)'])
    sns.set_style("whitegrid")
    ax = sns.boxplot(x="type", y="log(D2D(u,v))", data=scores, palette=['green', 'yellow', 'red'],
                     order=["TRUE_KEGG", "UNDIRECTED_KEGG", "FALSE_KEGG"], linewidth=1)
    ax.set_ylim([-1, 1])
    plt.setp(ax.lines, color=".1")
    ax.set(ylabel='log10(D2D score)')
    ax.set(xlabel='KEGG catalog')
    ax.set_xticklabels(['TRUE Orientations', 'UNDIRECTED', 'False Orientations'])
    ax.set_title('Single experiment D2D scores distribution for pathway {}'.format(KEGG_NAME))
    plt.savefig(r'C:\git\BNet21-D2D\results\human\{}\{}\SingleExperimentD2DscoresDistribution_{}.pdf'.format(
        KEGG_NAME, prior_weight_func_name, prior_weight_func_name
    ),
                format='pdf')
    plt.show()


if __name__ == '__main__':
    run_human_kegg_experiment(prior_weight_func_name='unweighted')
