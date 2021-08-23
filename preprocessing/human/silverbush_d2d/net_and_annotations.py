import copy
import json
import pandas as pd
import networkx as nx

from preprocessing.human.consts import RAW_HUMAN_NET_PATH, \
    GENERATED_HUMAN_NET_PATH, GENERATED_KEGG_ANNOTATIONS_PATH, \
    GENERATED_KEGG_TRUE_DIRECTION_ANNOTATIONS_PATH, GENERATED_KEGG_FALSE_DIRECTION_ANNOTATIONS_PATH


if __name__ == '__main__':
    experiment_dict = dict()

    net = pd.read_table(RAW_HUMAN_NET_PATH, header=None, names=['ID1', 'ID2', 'conf', 'direction', 'type'])
    edges_data = net.to_dict(orient='records')

    weighted_edges = list(map(
        lambda e: (str(e['ID1']), str(e['ID2']), e['conf']), edges_data))

    # Remove edges with weight 0
    weighted_edges = list(filter(lambda e_data: e_data[2] != 0, weighted_edges))

    g = nx.Graph()
    g.add_weighted_edges_from(weighted_edges)

    nx.write_gpickle(g, GENERATED_HUMAN_NET_PATH)

    true_annotations_data = list(filter(lambda e: e['type'] == 'TRUE_KEGG', edges_data))
    reverse_annotations_data = list(filter(lambda e: e['type'] == 'FALSE_KEGG', edges_data))

    true_kegg = list(map(lambda e: (str(e['ID1']), str(e['ID2'])), true_annotations_data))
    false_kegg = list(map(lambda e: (str(e['ID1']), str(e['ID2'])), reverse_annotations_data))
    reversed_false_kegg = list(map(lambda e: (e[1], e[0]), false_kegg))

    true_annotations = copy.copy(true_kegg)
    true_annotations.extend(reversed_false_kegg)

    with open(GENERATED_KEGG_ANNOTATIONS_PATH, 'w') as f:
        json.dump(list(set(true_annotations)), f)
    with open(GENERATED_KEGG_TRUE_DIRECTION_ANNOTATIONS_PATH, 'w') as f:
        json.dump(list(set(true_kegg)), f)
    with open(GENERATED_KEGG_FALSE_DIRECTION_ANNOTATIONS_PATH, 'w') as f:
        json.dump(list(set(false_kegg)), f)
