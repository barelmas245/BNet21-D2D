import os
import networkx as nx

from preprocessing.yeast.consts import RAW_YEAST_Y2H_PATH, GENERATED_YEAST_Y2H_NET_PATH


def get_y2h_union_network(src_path=RAW_YEAST_Y2H_PATH, dst_path=GENERATED_YEAST_Y2H_NET_PATH,
                          force=False):
    if os.path.isfile(dst_path) and not force:
        return nx.read_gpickle(dst_path)
    else:
        edges = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for line in data:
            line_data = line.replace('\n', '').split('\t')
            edges.append((line_data[0], line_data[1]))

        graph = nx.Graph()
        graph.add_edges_from(edges)
        nx.write_gpickle(graph, dst_path)
        return graph


if __name__ == '__main__':
    net = get_y2h_union_network(force=True)
