import json
from preprocessing.yeast.networks import get_undirected_net, enrich_net_with_directed_kpis, \
    direct_and_enrich_net_with
from preprocessing.yeast.annotations import get_kpis
from preprocessing.yeast.experiments import get_gene_expressions_data
from preprocessing.yeast.consts import GENERATED_MACISAAC_PDIS_PATH, GENERATED_EDGES_TO_DIRECT_PATH


def read_data(net_type='biogrid', undirected=True):
    if undirected:
        network = get_undirected_net(net_type=net_type)
        true_annotations_list = get_kpis(net_type=net_type)
        edges_to_direct = true_annotations_list
    else:
        network = enrich_net_with_directed_kpis(net_type=net_type)
        with open(str(GENERATED_MACISAAC_PDIS_PATH).format(net_type), 'r') as f:
            true_annotations_list = json.load(f)
        with open(str(GENERATED_EDGES_TO_DIRECT_PATH).format(net_type), 'r') as f:
            edges_to_direct = json.load(f)

    # Represent edges as tuples and not list
    directed_interactions = list(map(lambda e: tuple(e), true_annotations_list))
    if edges_to_direct:
        edges_to_direct = list(map(lambda e: tuple(e), edges_to_direct))
    gene_expressions = get_gene_expressions_data(net_type=net_type)

    return network, directed_interactions, gene_expressions, edges_to_direct
