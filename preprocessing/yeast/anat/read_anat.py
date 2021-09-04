import os
import networkx as nx

from preprocessing.yeast.consts import RAW_UNIPROT_ANAT_MAPPER_PATH, \
    RAW_YEAST_ANAT_PATH, GENERATED_ANAT_NET_PATH


def get_anat_yeast_network(src_path=RAW_YEAST_ANAT_PATH, dst_path=GENERATED_ANAT_NET_PATH,
                           force=False):
    if os.path.isfile(dst_path) and not force:
        return nx.read_gpickle(dst_path)
    else:
        mapping_gene_ids_to_name = dict()
        with open(RAW_UNIPROT_ANAT_MAPPER_PATH) as f:
            data = f.readlines()
        for line in data[1:]:
            line_data = line.replace(';\n', '').split('\t')
            gene_name = line_data[1].replace('_YEAST', '')
            gene_id = line_data[-1]
            mapping_gene_ids_to_name[gene_id] = gene_name

        edges = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for line in data:
            line_data = line.replace('\n', '').split('\t')
            gene_id1, gene_id2, weight = line_data[0], line_data[1], float(line_data[2])
            gene_name1 = mapping_gene_ids_to_name.get(gene_id1)
            gene_name2 = mapping_gene_ids_to_name.get(gene_id2)
            if gene_name1 and gene_name2:
                edges.append((str(gene_name1), str(gene_name2), weight))

        graph = nx.Graph()
        graph.add_weighted_edges_from(edges)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        nx.write_gpickle(graph, dst_path)
        return graph


if __name__ == '__main__':
    net = get_anat_yeast_network(force=True)
