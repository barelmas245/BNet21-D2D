import json
import networkx as nx

from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.y2h.read_y2h_union import get_y2h_union_network

from preprocessing.yeast.consts import GENERATED_YEAST_DIRECTED_NET_PATH,\
    GENERATED_BREITKREUTZ_ANNOTATIONS_PATH, GENERATED_MACISAAC_PDIS_PATH

PDIS = r'C:\git\BNet21-D2D\sources\yeast\macisacc_kpis_orfs_by_factor_p0.001_cons2.txt'
BIOGRID_PDI_MAPPER = r'C:\git\BNet21-D2D\sources\yeast\biogrid_pdis_mapper.tsv'
Y2H_PDI_MAPPER = r'C:\git\BNet21-D2D\sources\yeast\y2h_pdis_mapper.tsv'

KPIS = r'C:\git\BNet21-D2D\sources\yeast\generated\breitkeurtz_annotations.json'


def direct_and_enrich_net(net_type="biogrid"):
    net = get_undirected_net(net_type)

    with open(str(GENERATED_BREITKREUTZ_ANNOTATIONS_PATH).format(net_type), 'r') as f:
        kpis = json.load(f)
        kpis = list(map(lambda e: (e[0], e[1], 1), kpis))
    with open(str(GENERATED_MACISAAC_PDIS_PATH).format(net_type), 'r') as f:
        pdis = json.load(f)
        pdis = list(map(lambda e: (e[0], e[1], 1), pdis))

    for e in kpis:
        source, target, weight = e
        if net[source].get(target):
            net.remove_edge(source, target)

    net.add_weighted_edges_from(pdis)
    directed_graph = net.to_directed()
    directed_graph.add_weighted_edges_from(pdis)

    nx.write_gpickle(directed_graph, str(GENERATED_YEAST_DIRECTED_NET_PATH).format(net_type))

    return directed_graph


def get_undirected_net(net_type='biogrid'):
    if net_type == 'biogrid':
        net = get_biogrid_network(force=False)
        return net
    elif net_type == 'y2h':
        net = get_y2h_union_network(force=False)
        return net
    else:
        raise ValueError("Unsupported network type")


if __name__ == '__main__':
    direct_and_enrich_net(net_type='biogrid')
    direct_and_enrich_net(net_type='y2h')
