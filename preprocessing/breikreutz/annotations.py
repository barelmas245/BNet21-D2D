import networkx as nx
import json
import os

from preprocessing.biogrid.read_biogrid import get_biogrid_network
from preprocessing.breikreutz.experiments import get_gene_expressions_data
from preprocessing.consts import RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH, GENERATED_BREITKREUTZ_ANNOTATIONS_PATH


# def get_true_annotations(src_path=RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH,
#                          dst_path=GENERATED_BREITKREUTZ_ANNOTATIONS_PATH,
#                          filter_by_biogrid_net=True, force=False):
#     if os.path.isfile(dst_path) and not force:
#         with open(dst_path, 'r') as f:
#             true_annotations_list = json.load(f)
#             # Represent edges as tuples and not list
#             return list(map(lambda e: tuple(e), true_annotations_list))
#     else:
#         if filter_by_biogrid_net:
#             biogrid_net = get_biogrid_network()
#             biogrid_genes = biogrid_net.nodes
#
#         true_annotations_list = []
#         with open(src_path, 'r') as f:
#             data = f.readlines()
#         for entry in data:
#             _, src_gene, _, _, dst_genes = entry.replace('\n', '').split('\t')
#             for dst_gene in dst_genes.split(','):
#                 if filter_by_biogrid_net and (src_gene not in biogrid_genes or dst_gene not in biogrid_genes):
#                     continue
#                 # Do not include both direction edges
#                 if (dst_gene, src_gene) in true_annotations_list:
#                     continue
#                 true_annotations_list.append((src_gene, dst_gene))
#
#         with open(dst_path, 'w') as f:
#             json.dump(true_annotations_list, f)
#
#         return true_annotations_list


def get_true_annotations(src_path=RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH,
                         dst_path=GENERATED_BREITKREUTZ_ANNOTATIONS_PATH,
                         filter_by_biogrid_net=True, force=False):
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            true_annotations_list = json.load(f)
            # Represent edges as tuples and not list
            return list(map(lambda e: tuple(e), true_annotations_list))
    else:
        biogrid_net = get_biogrid_network()
        biogrid_genes = biogrid_net.nodes
        experiments_dict = get_gene_expressions_data()
        all_relevant_edges = set()
        for source_gene in experiments_dict:
            for target_gene in experiments_dict[source_gene]:
                path = nx.shortest_path(biogrid_net, source=source_gene, target=target_gene)
                edges_path = nx.utils.pairwise(path)
                all_relevant_edges = all_relevant_edges.union(list(edges_path))

        true_annotations_list = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for entry in data[1:]:
            gene1, sys_name1, gene2, sys_name2, assay, method, action, modification, pub_source, ref, note = entry.replace('\n', '').split('\t')
            if filter_by_biogrid_net and (gene1 not in biogrid_genes or gene2 not in biogrid_genes):
                continue

            if method == 'high-throughput' and pub_source == 'BioGRID':
                if action == 'Bait-Hit':
                    edge = (gene1, gene2)
                elif action == 'Hit-Bait':
                    edge = (gene2, gene1)
                else:
                    raise Exception()

            # Do not include both direction edges
            if edge in true_annotations_list:
                continue

            reversed_edge = (edge[1], edge[0])
            if edge in all_relevant_edges or reversed_edge in all_relevant_edges:
                true_annotations_list.append(edge)

        with open(dst_path, 'w') as f:
            json.dump(true_annotations_list, f)

        return true_annotations_list



if __name__ == '__main__':
    true_annotations = get_true_annotations(force=True)
