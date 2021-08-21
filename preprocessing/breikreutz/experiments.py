import pandas as pd
import numpy as np
import json
import os

from preprocessing.biogrid.read_biogrid import get_biogrid_network
from preprocessing.consts import RAW_BREITKREUTZ_EXPRESSIONS_DATA_PATH, GENERATED_BREITKREUTZ_EXPRESSIONS_PATH


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
        for i in raw_data:
            if filter_by_biogrid_net and i not in biogrid_genes:
                continue
            targets_dict = raw_data[i].to_dict()
            targets_dict = {key: value for key, value in targets_dict.items() if abs(value) >= np.log2(1.7)}
            experiments_dict[i] = dict([(key, val) for key, val in targets_dict.items() if key in biogrid_genes]) if filter_by_biogrid_net else targets_dict

        with open(dst_path, 'w') as f:
            json.dump(experiments_dict, f)

        return experiments_dict


if __name__ == '__main__':
    gene_expressions = get_gene_expressions_data(force=True)
