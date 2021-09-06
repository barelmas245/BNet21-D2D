import json
import networkx as nx

from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.y2h.read_y2h_union import get_y2h_union_network
from preprocessing.yeast.anat.read_anat import get_anat_yeast_network

from preprocessing.yeast.consts import GENERATED_YEAST_DIRECTED_NET_PATH,\
    GENERATED_BREITKREUTZ_ANNOTATIONS_PATH, GENERATED_MACISAAC_PDIS_PATH, \
    GENERATED_EDGES_TO_DIRECT_PATH


def enrich_net_with_directed_kpis(net_type="biogrid"):
    net = get_undirected_net(net_type)

    with open(str(GENERATED_BREITKREUTZ_ANNOTATIONS_PATH).format(net_type), 'r') as f:
        kpis = json.load(f)
        kpis = list(map(lambda e: (e[0], e[1], 1), kpis))
    with open(str(GENERATED_MACISAAC_PDIS_PATH).format(net_type), 'r') as f:
        pdis = json.load(f)
        pdis = list(map(lambda e: (e[0], e[1], 1), pdis))

    for e in kpis:
        source, target, weight = e
        if source in net and net[source].get(target):
            net.remove_edge(source, target)

    net.add_weighted_edges_from(pdis)
    edges_to_direct = list(net.edges)
    directed_graph = net.to_directed()
    directed_graph.add_weighted_edges_from(kpis)

    nx.write_gpickle(directed_graph, str(GENERATED_YEAST_DIRECTED_NET_PATH).format(net_type))

    with open(str(GENERATED_EDGES_TO_DIRECT_PATH).format(net_type), 'w') as f:
        f.write(json.dumps(list(map(lambda x: (x[0], x[1]), edges_to_direct))))

    return directed_graph


def direct_and_enrich_net_with(net_type="biogrid", directed_edges=None):
    net = get_undirected_net(net_type)

    weighted_directed_edges = []
    if directed_edges:
        for e in directed_edges:
            source, target = e
            if source in net and net[source].get(target):
                weighted_directed_edges.append((source, target, net[source][target]["weight"]))
                net.remove_edge(source, target)
            else:
                directed_edges.remove(e)

    directed_graph = net.to_directed()
    if weighted_directed_edges:
        directed_graph.add_weighted_edges_from(weighted_directed_edges)

    nx.write_gpickle(directed_graph, str(GENERATED_YEAST_DIRECTED_NET_PATH).format(net_type))

    return directed_graph


def get_undirected_net(net_type='biogrid'):
    if net_type == 'biogrid':
        net = get_biogrid_network(force=False)
    elif net_type == 'y2h':
        net = get_y2h_union_network(force=False)
    elif net_type == 'anat':
        net = get_anat_yeast_network(force=False)
    else:
        raise ValueError("Unsupported network type")

    with open(str(GENERATED_BREITKREUTZ_ANNOTATIONS_PATH).format(net_type), 'r') as f:
        kpis = json.load(f)
        weighted_kpis = []
        for e in kpis:
            source, target = e
            if source in net and net[source].get(target):
                weighted_kpis.append((source, target, net[source][target]["weight"]))
            else:
                weighted_kpis.append((source, target, 1))

    net.add_weighted_edges_from(weighted_kpis)
    return net


if __name__ == '__main__':
    enrich_net_with_directed_kpis(net_type='biogrid')
    enrich_net_with_directed_kpis(net_type='y2h')
