import pandas as pd
import json

from preprocessing.human.consts import RAW_KEGG_SOURCES_DATA_PATH, RAW_KEGG_TARGETS_DATA_PATH,\
    GENERATED_EXPERIMENT_PATH


if __name__ == '__main__':
    experiment_dict = dict()

    sources = list(pd.read_table(RAW_KEGG_SOURCES_DATA_PATH, header=None, names=['Experiment', 'ID'])['ID'].values)
    terminals = list(pd.read_table(RAW_KEGG_TARGETS_DATA_PATH, header=None, names=['Experiment', 'ID'])['ID'].values)

    experiment_dict[','.join(map(lambda x: str(x), sources))] = dict(
        map(lambda target: (str(target), 1), terminals))
    with open(GENERATED_EXPERIMENT_PATH, 'w') as f:
        json.dump(experiment_dict, f, indent=2)
