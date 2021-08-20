import pandas as pd
import json
import os

from network.biogrid.read_biogrid import get_biogrid_network

EXPERIMENTS_EXPRESSIONS_DATA_PATH = r'C:\git\BNet21-D2D\sources\Lorenzo data\1-s2.0-S0092867414003420-mmc3\Data_S1.cdt'
EXPRESSIONS_PATH = r'C:\git\BNet21-D2D\sources\generated\breitkeurtz_expressions.json'


def get_gene_expressions_data(filter_by_biogrid_net=True, force=False):
    if os.path.isfile(EXPRESSIONS_PATH) and not force:
        with open(EXPRESSIONS_PATH, 'r') as f:
            return json.load(f)
    else:
        if filter_by_biogrid_net:
            biogrid_net = get_biogrid_network()
            biogrid_genes = biogrid_net.nodes

        raw_data = pd.read_csv(
            EXPERIMENTS_EXPRESSIONS_DATA_PATH, sep="\t", index_col=3, skiprows=[0, 1, 2, 4, 5, 6])
        raw_data.drop(raw_data.columns[:6], axis=1, inplace=True)

        experiments_dict = dict()
        for i in raw_data:
            if filter_by_biogrid_net and i not in biogrid_genes:
                continue
            targets_dict = raw_data[i].to_dict()
            experiments_dict[i] = dict([(key, val) for key, val in targets_dict.items() if key in biogrid_genes]) if filter_by_biogrid_net else targets_dict

        with open(EXPRESSIONS_PATH, 'w') as f:
            json.dump(experiments_dict, f)

        return experiments_dict


if __name__ == '__main__':
    gene_expressions = get_gene_expressions_data(force=True)
