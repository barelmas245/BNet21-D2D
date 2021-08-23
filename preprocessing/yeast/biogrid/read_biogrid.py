import os
import numpy as np
import networkx as nx

from preprocessing.yeast.biogrid.conf import BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES, ONLY_PHYSICAL
from preprocessing.yeast.consts import RAW_YEAST_BIOGRID_PATH, GENERATED_YEAST_BIOGRID_NET_PATH


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


def calculate_interaction_score(interaction_experiments):
    return 1 - np.prod(list(map(
        lambda exp: 1 - BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES[exp], list(set(interaction_experiments)))))


def read_biogrod_data(path=RAW_YEAST_BIOGRID_PATH, only_physical=ONLY_PHYSICAL):
    with open(path, 'r') as f:
        data = f.readlines()
    all_interactions = list(map(lambda entry_line: BioGridEntryData(entry_line.replace('\n', '')), data[1:]))
    return list(filter(lambda entry: entry.exp_system_type == 'physical', all_interactions)) if only_physical else all_interactions


def get_biogrid_network(src_path=RAW_YEAST_BIOGRID_PATH, dst_path=GENERATED_YEAST_BIOGRID_NET_PATH, force=False):
    if os.path.isfile(dst_path) and not force:
        return nx.read_gpickle(dst_path)
    else:
        data = read_biogrod_data(src_path)

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

        nx.write_gpickle(g, dst_path)

        return g


if __name__ == '__main__':
    net = get_biogrid_network(force=True)
