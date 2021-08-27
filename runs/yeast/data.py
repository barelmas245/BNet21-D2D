from preprocessing.yeast.networks import get_undirected_net, direct_and_enrich_net
from preprocessing.yeast.experiments import get_gene_expressions_data
from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.y2h.read_y2h_union import get_y2h_union_network


from preprocessing.yeast.consts import GENERATED_MACISAAC_PDIS_PATH


def read_data(net_type='biogrid', undirected=True):
    if undirected:
        network = get_undirected_net(net_type=net_type)
    else:
        network = direct_and_enrich_net(net_type=net_type)
    gene_expressions = get_gene_expressions_data(net_type=net_type)

    import json
    with open(str(GENERATED_MACISAAC_PDIS_PATH).format(net_type), 'r') as f:
        true_annotations_list = json.load(f)
        # Represent edges as tuples and not list
        directed_interactions = list(map(lambda e: tuple(e), true_annotations_list))
    return network, directed_interactions, gene_expressions
