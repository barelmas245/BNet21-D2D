import json
import os

from network.biogrid.read_biogrid import get_biogrid_network

TRUE_ANNOTATIONS_DATA_PATH = r'C:\git\BNet21-D2D\sources\true_annotations\Breitkreutz data.txt'
TRUE_ANNOTATIONS_PATH = r'C:\git\BNet21-D2D\sources\generated\true_annotations.json'


def get_true_annotations(filter_by_biogrid_net=True, force=False):
    if os.path.isfile(TRUE_ANNOTATIONS_PATH) and not force:
        with open(TRUE_ANNOTATIONS_PATH, 'r') as f:
            return json.load(f)
    else:
        if filter_by_biogrid_net:
            biogrid_net = get_biogrid_network()
            biogrid_genes = biogrid_net.nodes

        true_annotations_list = []
        with open(TRUE_ANNOTATIONS_DATA_PATH, 'r') as f:
            data = f.readlines()
        for entry in data:
            _, src_gene, _, _, dst_genes = entry.replace('\n', '').split('\t')
            for dst_gene in dst_genes.split(','):
                if filter_by_biogrid_net and (src_gene not in biogrid_genes or dst_gene not in biogrid_genes):
                    continue
                true_annotations_list.append((src_gene, dst_gene))

        with open(TRUE_ANNOTATIONS_PATH, 'w') as f:
            json.dump(true_annotations_list, f)

        return true_annotations_list


if __name__ == '__main__':
    true_annotations = get_true_annotations(force=True)
