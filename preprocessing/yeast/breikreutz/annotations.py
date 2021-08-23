import json
import os

from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.consts import RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH, GENERATED_BREITKREUTZ_ANNOTATIONS_PATH


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
            true_annotations_list.append(edge)

        with open(dst_path, 'w') as f:
            json.dump(true_annotations_list, f)

        return true_annotations_list


if __name__ == '__main__':
    true_annotations = get_true_annotations(force=True)
