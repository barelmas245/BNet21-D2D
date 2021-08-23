from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.experiments import get_gene_expressions_data
from preprocessing.yeast.annotations import get_true_annotations


def read_data():
    network = get_biogrid_network()
    gene_expressions = get_gene_expressions_data()
    directed_interactions = get_true_annotations()
    return network, directed_interactions, gene_expressions
