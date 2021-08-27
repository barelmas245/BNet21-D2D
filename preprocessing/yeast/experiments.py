import pandas as pd
import numpy as np
import json
import os

from preprocessing.yeast.networks import get_undirected_net
from preprocessing.yeast.consts import RAW_HOLSTEGE_EXPRESSIONS_DATA_PATH, GENERATED_HOLSTEGE_EXPRESSIONS_PATH


def get_gene_expressions_data(src_path=RAW_HOLSTEGE_EXPRESSIONS_DATA_PATH,
                              dst_path=GENERATED_HOLSTEGE_EXPRESSIONS_PATH,
                              net_type='biogrid', filter_by_net=True, force=False):
    dst_path = str(dst_path).format(net_type)
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            return json.load(f)
    else:
        if filter_by_net:
            net = get_undirected_net(net_type)
            if net_type == 'biogrid':
                index_col = 3
            elif net_type == 'y2h':
                index_col = 2
            else:
                raise ValueError("Unsupported network type")
            genes = net.nodes

        skip_rows = list(range(7))
        skip_rows.remove(index_col)
        raw_data = pd.read_csv(
            src_path, sep="\t", index_col=index_col, skiprows=skip_rows)
        raw_data.drop(raw_data.columns[:6], axis=1, inplace=True)

        experiments_dict = dict()
        all_sources = raw_data.columns
        all_targets = raw_data.index
        for src_gene in raw_data:
            if (filter_by_net and src_gene not in genes) or src_gene in all_targets:
                continue
            targets_dict = raw_data[src_gene].to_dict()
            # Filter out targets which has no significant expression or has negative score
            targets_dict = {key: value for key, value in targets_dict.items() if value <= - np.log2(1.7)}
            # Filter out targets which are also in the sources
            targets_dict = {key: value for key, value in targets_dict.items() if key not in all_sources}

            targets_dict = dict([(key, val) for key, val in targets_dict.items() if key in genes]) if filter_by_net else targets_dict
            if targets_dict != {}:
                experiments_dict[src_gene] = targets_dict

        with open(dst_path, 'w') as f:
            json.dump(experiments_dict, f, indent=2)

        return experiments_dict


if __name__ == '__main__':
    get_gene_expressions_data(net_type='biogrid', force=True)
    get_gene_expressions_data(net_type='y2h', force=True)
