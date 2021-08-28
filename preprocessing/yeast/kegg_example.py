import json
import networkx as nx
from pykml import parser

from preprocessing.yeast.consts import YEAST_SOURCES_DIR, YEAST_GENERATED_SOURCES_DIR

KEGG_FOLDER = YEAST_SOURCES_DIR / 'KEGG'
KEGG_GENERATED_FOLDER = YEAST_GENERATED_SOURCES_DIR / 'KEGG'


if __name__ == '__main__':
    all_kegg_annotations = []
    all_experiments = dict()
    for kegg_file_path in KEGG_FOLDER.iterdir():
        if kegg_file_path.is_file():
            with open(kegg_file_path, 'r') as f:
                data = f.read()
            root = parser.fromstring(data)
            all_elements = root.getchildren()

            all_genes = dict()
            all_relations = []
            for e in all_elements:
                if e.tag == 'entry' and e.attrib['type'] == 'gene':
                    gene_names = e.attrib['name'].replace('sce:', '').split(' ')
                    all_genes[e.attrib['id']] = gene_names
                if e.tag == 'relation':
                    relation = (e.attrib['entry1'], e.attrib['entry2'])
                    all_relations.append(relation)

            edges = []
            for r in all_relations:
                mapped_gene1_list = all_genes.get(r[0])
                mapped_gene2_list = all_genes.get(r[1])
                if mapped_gene1_list and mapped_gene2_list:
                    for gene1 in mapped_gene1_list:
                        for gene2 in mapped_gene2_list:
                            edges.append((gene1, gene2))

            g = nx.Graph()
            g.add_edges_from(edges)
            nx.write_gpickle(g, KEGG_GENERATED_FOLDER / kegg_file_path.stem / 'net.gpickle')

            true_annotations = edges
            with open(KEGG_GENERATED_FOLDER / kegg_file_path.stem / 'kegg_annotations.json', 'w') as f:
                json.dump(list(set(true_annotations)), f)

            directed_g = nx.DiGraph()
            directed_g.add_edges_from(edges)
            sources = list(filter(lambda x: directed_g.in_degree(x) == 0, g.nodes))
            targets = list(filter(lambda x: directed_g.out_degree(x) == 0, g.nodes))

            experiment_dict = dict()
            experiment_dict[','.join(map(lambda x: str(x), sources))] = dict(
                map(lambda target: (str(target), 1), targets))
            with open(KEGG_GENERATED_FOLDER / kegg_file_path.stem / 'experiment.json', 'w') as f:
                json.dump(experiment_dict, f, indent=2)

            all_kegg_annotations.extend(true_annotations)
            all_experiments.update(experiment_dict)

    all_kegg_annotations = set(all_kegg_annotations)

    g = nx.Graph()
    g.add_edges_from(all_kegg_annotations)
    nx.write_gpickle(g, KEGG_GENERATED_FOLDER / 'net.gpickle')

    all_kegg_annotations = all_kegg_annotations.difference(map(
        lambda e: (e[1], e[0]), all_kegg_annotations))

    with open(KEGG_GENERATED_FOLDER / 'all_kegg_annotations.json', 'w') as f:
        json.dump(list(all_kegg_annotations), f)
    with open(KEGG_GENERATED_FOLDER / 'all_experiments.json', 'w') as f:
        json.dump(all_experiments, f, indent=2)
