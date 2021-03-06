import pandas as pd
import numpy as np
import json
import os

from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.consts import RAW_HOLSTEGE_EXPRESSIONS_DATA_PATH, GENERATED_HOLSTEGE_EXPRESSIONS_PATH

SOURCES = [
    'SLN1',
    'YCK1',
    'YCK2',
    'SHO1',
    'MF(ALPHA)2',
    'MID2',
    'RAS2',
    'GPR1',
    'BCY1',
    'STE50',
    'MSB2',
    'SIN3',
    'RGA1',
    'RGA2',
    'ARR4'
]

TARGETS = [
    'CDC42',
    'HOG1',
    'STE7',
    'STE20',
    'DIG2',
    'DIG1',
    'PBS2',
    'FUS3',
    'STE5',
    'GPA1',
    'FKS2',
    'FUS1',
    'STE12',
    'SWI4'
]

TARGETS_BAREL = [
    'FUS1',
    'FAR1',
    'GPD1',
    'CTT1',
    'GRE2'
]


def get_gene_expressions_data(src_path=RAW_HOLSTEGE_EXPRESSIONS_DATA_PATH,
                              dst_path=GENERATED_HOLSTEGE_EXPRESSIONS_PATH,
                              filter_by_biogrid_net=True, force=False):
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            return json.load(f)
    else:
        if filter_by_biogrid_net:
            biogrid_net = get_biogrid_network()
            biogrid_genes = biogrid_net.nodes

        raw_data = pd.read_csv(
            src_path, sep="\t", index_col=3, skiprows=[0, 1, 2, 4, 5, 6])
        raw_data.drop(raw_data.columns[:6], axis=1, inplace=True)

        experiments_dict = dict()
        all_sources = raw_data.columns
        all_targets = raw_data.index

        final_targets = {}
        for src_gene in raw_data:
            if (filter_by_biogrid_net and src_gene not in biogrid_genes) or src_gene in all_targets:
                continue
            targets_dict = raw_data[src_gene].to_dict()
            # Filter out targets which has no significant expression or has negative score
            targets_dict = {key: value for key, value in targets_dict.items() if value <= - np.log2(1.7)}
            # Filter out targets which are also in the sources
            targets_dict = {key: value for key, value in targets_dict.items() if key not in all_sources}

            targets_dict = dict([(key, val) for key, val in targets_dict.items() if key in biogrid_genes]) if filter_by_biogrid_net else targets_dict
            if targets_dict != {}:
                experiments_dict[src_gene] = targets_dict

                for target in targets_dict:
                    if target in final_targets:
                        final_targets[target].append(targets_dict[target])
                    else:
                        final_targets[target] = [targets_dict[target]]

        final_sources = list(experiments_dict.keys())
        with open(r'C:\git\BNet21-D2D\results\single_unified_experiment\sources.txt', 'w') as f:
            lines = list(map(lambda source: 'KEGG\t' + source + '\n', final_sources))
            f.writelines(lines)

        with open(r'C:\git\BNet21-D2D\results\single_unified_experiment\targets.txt', 'w') as f:
            lines = list(map(lambda target: 'KEGG\t' + target + '\t' + str(np.mean(final_targets[target])) + '\n',
                             final_targets))
            f.writelines(lines)

        return experiments_dict


if __name__ == '__main__':
    gene_expressions = get_gene_expressions_data(force=True)
