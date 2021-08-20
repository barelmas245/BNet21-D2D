import os
import json
import pickle
import numpy as np
import networkx as nx

from network.biogrid.conf import BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES, ONLY_PHYSICAL

YEAST_BIOGRID_TXT_PATH = r'C:\git\BNet21-D2D\sources\BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-4.4.199.tab3.txt'
BIOGRID_NET_PATH = r'C:\git\BNet21-D2D\sources\generated\biogrid_net.gpickle'

# YEAST_BIOGRID_DATA_PATH = r'C:\git\BNet21-D2D\sources\yeast_biogrid_data.json'
# YEAST_BIOGRID_PROTEINS_IDS_PATH = r'C:\git\BNet21-D2D\sources\generated\biogrid_yeast_proteins_ids.json'
# YEAST_BIOGRID_PROTEINS_IDS_MAP_TO_UNIPROT_PATH = r'C:\git\BNet21-D2D\sources\generated\map_biogrid_ids_to_uniprot.json'
# YEAST_BIOGRID_FAILED_IDS_PATH = r'C:\git\BNet21-D2D\sources\generated\failed_ids.json'
#
# YEAST_BIOGRID_DISTINCT_INTERACTIONS_PATH = r'C:\git\BNet21-D2D\sources\generated\biogrid_interactions.pickle'


class BioGridInteractorData(object):
    def __init__(self, entrez_gene, biogrid_id, systematic_name, symbol, synonyms,
                 organism_id, organism_name, swiss_prot_accessions, trembl_accessions, refseq_accessions):
        self.entrez_gene = entrez_gene
        self.biogrid_id = biogrid_id
        self.systematic_name = systematic_name
        self.symbol = symbol
        self.synonyms = synonyms.split('|') if synonyms else synonyms
        self.organism_id = organism_id
        self.organism_name = organism_name
        self.swiss_prot_accessions = swiss_prot_accessions
        self.trembl_accessions = trembl_accessions
        self.refseq_accessions = refseq_accessions

    @property
    def all_symbols(self):
        symbols = [self.symbol]
        symbols.extend(self.synonyms)
        return symbols


class BioGridEntryData(object):
    def __init__(self, data_line):
        data_list = data_line.split('\t')
        self.biogrid_interaction_id = self._set_biogrid_attr(data_list[0])

        self.exp_system = self._set_biogrid_attr(data_list[11])
        self.exp_system_type = self._set_biogrid_attr(data_list[12])
        self.author = self._set_biogrid_attr(data_list[13])
        self.pub_source = self._set_biogrid_attr(data_list[14])

        self.throughput = self._set_biogrid_attr(data_list[17])
        self.score = self._set_score(data_list[18])
        self.modification = self._set_biogrid_attr(data_list[19])
        self.qualifications = self._set_biogrid_attr(data_list[20])
        self.tags = self._set_biogrid_attr(data_list[21])
        self.source_database = self._set_biogrid_attr(data_list[22])

        self.ontology_term_ids = self._set_biogrid_attr(data_list[29])
        self.ontology_term_names = self._set_biogrid_attr(data_list[30])
        self.ontology_term_categories = self._set_biogrid_attr(data_list[31])
        self.ontology_term_qualifier_ids = self._set_biogrid_attr(data_list[32])
        self.ontology_term_qualifier_names = self._set_biogrid_attr(data_list[33])
        self.ontology_term_types = self._set_biogrid_attr(data_list[34])

        self.interactor_a = BioGridInteractorData(
            entrez_gene=self._set_biogrid_attr(data_list[1]),
            biogrid_id=self._set_biogrid_attr(data_list[3]),
            systematic_name=self._set_biogrid_attr(data_list[5]),
            symbol=self._set_biogrid_attr(data_list[7]),
            synonyms=self._set_biogrid_attr(data_list[9]),
            organism_id=self._set_biogrid_attr(data_list[15]),
            organism_name=self._set_biogrid_attr(data_list[35]),
            swiss_prot_accessions=self._set_biogrid_attr(data_list[23]),
            trembl_accessions=self._set_biogrid_attr(data_list[24]),
            refseq_accessions=self._set_biogrid_attr(data_list[25])
        )

        self.interactor_b = BioGridInteractorData(
            entrez_gene=self._set_biogrid_attr(data_list[2]),
            biogrid_id=self._set_biogrid_attr(data_list[4]),
            systematic_name=self._set_biogrid_attr(data_list[6]),
            symbol=self._set_biogrid_attr(data_list[8]),
            synonyms=self._set_biogrid_attr(data_list[10]),
            organism_id=self._set_biogrid_attr(data_list[16]),
            organism_name=self._set_biogrid_attr(data_list[36]),
            swiss_prot_accessions=self._set_biogrid_attr(data_list[26]),
            trembl_accessions=self._set_biogrid_attr(data_list[27]),
            refseq_accessions=self._set_biogrid_attr(data_list[28])
        )

    @staticmethod
    def _set_biogrid_attr(attr_str):
        return None if attr_str == '-' else attr_str

    @staticmethod
    def _set_score(score_str):
        return None if score_str == '-' else float(score_str)


# class ProteinInteraction(object):
#     def __init__(self, interactor_a, interactor_b, breikreutz=[]):
#         self.interactor_a = interactor_a
#         self.interactor_b = interactor_b
#         self.breikreutz = breikreutz
#
#         self.score = None
#         self.calculate_score()
#
#     @property
#     def relevant_experiments(self):
#         return list(filter(lambda exp: exp in BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES, self.breikreutz))
#
#     def calculate_score(self):
#         self.score = 1 - np.prod(list(map(
#             lambda exp: 1 - BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES[exp], list(set(self.breikreutz)))))

def calculate_interaction_score(interaction_experiments):
    return 1 - np.prod(list(map(
        lambda exp: 1 - BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES[exp], list(set(interaction_experiments)))))


def read_biogrod_data(only_physical=ONLY_PHYSICAL):
    with open(YEAST_BIOGRID_TXT_PATH, 'r') as f:
        data = f.readlines()
    all_interactions = list(map(lambda entry_line: BioGridEntryData(entry_line.replace('\n', '')), data[1:]))
    return list(filter(lambda entry: entry.exp_system_type == 'physical', all_interactions)) if only_physical else all_interactions


def get_biogrid_network(force=False):
    if os.path.isfile(BIOGRID_NET_PATH) and not force:
        return nx.read_gpickle(BIOGRID_NET_PATH)
    else:
        data = read_biogrod_data()

        distinct_interactions_dict = dict()
        for entry in data:
            interaction_tuple = (entry.interactor_a.symbol, entry.interactor_b.symbol)
            if interaction_tuple in distinct_interactions_dict:
                distinct_interactions_dict[interaction_tuple].add(entry.exp_system)
            else:
                distinct_interactions_dict[interaction_tuple] = {entry.exp_system}

        weighted_edges = list(map(
            lambda i: (i[0], i[1], calculate_interaction_score(distinct_interactions_dict[i])),
            distinct_interactions_dict))
        # Remove edges with weight 0
        weighted_edges = list(filter(lambda e_data: e_data[2] != 0, weighted_edges))

        g = nx.Graph()
        g.add_weighted_edges_from(weighted_edges)
        g.remove_edges_from(nx.selfloop_edges(g))
        g.remove_nodes_from(list(nx.isolates(g)))

        largest_component = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_component).copy()

        nx.write_gpickle(g, BIOGRID_NET_PATH)

        return g


# def get_all_proteins_biogrid_ids(biogrid_data):
#     a_ids = set(list(map(lambda biogrid_entry: biogrid_entry.interactor_a.biogrid_id, biogrid_data)))
#     b_ids = set(list(map(lambda biogrid_entry: biogrid_entry.interactor_a.biogrid_id, biogrid_data)))
#     return a_ids.union(b_ids)


if __name__ == '__main__':
    net = get_biogrid_network(force=True)
    # data = read_biogrod_data()
    #
    # distinct_interactions_dict = dict()
    # for entry in data:
    #     interaction_tuple = (entry.interactor_a.symbol, entry.interactor_b.symbol)
    #     if interaction_tuple in distinct_interactions_dict:
    #         distinct_interactions_dict[interaction_tuple].add(entry.exp_system)
    #     else:
    #         distinct_interactions_dict[interaction_tuple] = {entry.exp_system}
    #
    # g = nx.Graph()
    # g.add_weighted_edges_from(list(map(
    #     lambda i: (i[0], i[1], calculate_interaction_score(distinct_interactions_dict[i])),
    #     distinct_interactions_dict)))
    #
    # with open(BIOGRID_NET_PATH, 'wb') as f:
    #     pickle.dump(g, f)

    # all_ids = get_all_proteins_biogrid_ids(data)
    # with open(YEAST_BIOGRID_PROTEINS_IDS_PATH, 'w') as f:
    #     f.write(json.dumps(list(all_ids)))

    # if os.path.isfile(YEAST_BIOGRID_PROTEINS_IDS_MAP_TO_UNIPROT_PATH):
    #     with open(YEAST_BIOGRID_PROTEINS_IDS_MAP_TO_UNIPROT_PATH, 'r') as f:
    #         map_to_uniprot = json.load(f)
    # else:
    #     map_to_uniprot = map_identifiers(all_ids, from_id=BIOGRID_ID, to_id=UNIPROT_ID)
    #     with open(YEAST_BIOGRID_PROTEINS_IDS_MAP_TO_UNIPROT_PATH, 'w') as f:
    #         f.write(json.dumps(map_to_uniprot))
    #
    # failed_set = set()
    # distinct_interactions_dict = dict()
    # for entry in data:
    #     uniprot_a = map_to_uniprot.get(entry.interactor_a.biogrid_id)
    #     uniprot_b = map_to_uniprot.get(entry.interactor_b.biogrid_id)
    #     if uniprot_a is None:
    #         failed_set.add(entry.interactor_a.biogrid_id)
    #         # print(f"Failed to map {entry.interactor_a.biogrid_id}")
    #         continue
    #     if uniprot_b is None:
    #         failed_set.add(entry.interactor_b.biogrid_id)
    #         # print(f"Failed to map {entry.interactor_b.biogrid_id}")
    #         continue
    #     interaction_tuple = (uniprot_a, uniprot_b)
    #     if interaction_tuple in distinct_interactions_dict:
    #         distinct_interactions_dict[interaction_tuple].append(entry.exp_system)
    #     else:
    #         distinct_interactions_dict[interaction_tuple] = [entry.exp_system]
    # with open(YEAST_BIOGRID_FAILED_IDS_PATH, 'w') as f:
    #     f.write(json.dumps(list(failed_set)))
    #
    # all_distinct_interactions = list(map(lambda i: ProteinInteraction(i[0], i[1], distinct_interactions_dict[i]),
    #                                      distinct_interactions_dict))
    # with open(YEAST_BIOGRID_DISTINCT_INTERACTIONS_PATH, 'wb') as f:
    #     pickle.dump(all_distinct_interactions, f)
