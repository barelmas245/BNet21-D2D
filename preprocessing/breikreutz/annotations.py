import json
import os

from preprocessing.biogrid.read_biogrid import get_biogrid_network
from preprocessing.consts import RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH, GENERATED_BREITKREUTZ_ANNOTATIONS_PATH


def get_true_annotations(src_path=RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH,
                         dst_path=GENERATED_BREITKREUTZ_ANNOTATIONS_PATH,
                         filter_by_biogrid_net=True, force=False):
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            true_annotations_list = json.load(f)
            # Represent edges as tuples and not list
            return list(map(lambda e: tuple(e), true_annotations_list))
    else:
        if filter_by_biogrid_net:
            biogrid_net = get_biogrid_network()
            biogrid_genes = biogrid_net.nodes

        true_annotations_list = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for entry in data:
            _, src_gene, _, _, dst_genes = entry.replace('\n', '').split('\t')
            for dst_gene in dst_genes.split(','):
                if filter_by_biogrid_net and (src_gene not in biogrid_genes or dst_gene not in biogrid_genes):
                    continue
                # Do not include both direction edges
                if (dst_gene, src_gene) in true_annotations_list:
                    continue
                true_annotations_list.append((src_gene, dst_gene))

        with open(dst_path, 'w') as f:
            json.dump(true_annotations_list, f)

        return true_annotations_list


if __name__ == '__main__':
    true_annotations = get_true_annotations(force=True)
