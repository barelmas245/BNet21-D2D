import pandas as pd
import numpy as np
import json
import os

from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.consts import RAW_BREITKREUTZ_EXPRESSIONS_DATA_PATH, GENERATED_BREITKREUTZ_EXPRESSIONS_PATH


def get_gene_expressions_data(src_path=RAW_BREITKREUTZ_EXPRESSIONS_DATA_PATH,
                              dst_path=GENERATED_BREITKREUTZ_EXPRESSIONS_PATH,
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

        with open(dst_path, 'w') as f:
            json.dump(experiments_dict, f, indent=2)

        return experiments_dict


if __name__ == '__main__':
    gene_expressions = get_gene_expressions_data(force=True)
