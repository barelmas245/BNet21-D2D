import json
import networkx as nx
from pykml import parser
import pandas as pd

from preprocessing.yeast.consts import YEAST_SOURCES_DIR, YEAST_GENERATED_SOURCES_DIR

KEGG_FOLDER = YEAST_SOURCES_DIR / 'KEGG'
KEGG_GENERATED_FOLDER = YEAST_GENERATED_SOURCES_DIR / 'KEGG'


if __name__ == '__main__':
    all_kegg_annotations = []
    all_experiments = dict()
    reconstructed_experiments = dict()

    reconstructed_nodes = pd.read_csv(
        r'C:\git\BNet21-D2D\sources\yeast\generated\KEGG\reconstructed_net_ANAT_nodes.csv')
    reconstructed_edges = pd.read_csv(
        r'C:\git\BNet21-D2D\sources\yeast\generated\KEGG\reconstructed_net_ANAT_edges.csv')
    nodes_map = dict()
    for node_data in reconstructed_nodes.values:
        name = node_data[0]
        ref_name = node_data[-1]
        nodes_map[ref_name] = name
    edges = []
    for e_data in reconstructed_edges.values:
        weight = e_data[0]
        gene1 = nodes_map[e_data[1]]
        gene2 = nodes_map[e_data[2]]
        edges.append((gene1, gene2, weight))
    reconstructed_g = nx.Graph()
    reconstructed_g.add_weighted_edges_from(edges)
    nx.write_gpickle(reconstructed_g, r'C:\git\BNet21-D2D\sources\yeast\generated\KEGG\reconstructed_net.gpickle')

    for kegg_file_path in KEGG_FOLDER.iterdir():
        kegg_folder = KEGG_GENERATED_FOLDER / kegg_file_path.stem
        if not kegg_folder.is_dir():
            kegg_folder.mkdir()

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

            reconstructed_sources = list(filter(lambda x: x in reconstructed_g, sources))
            reconstructed_targets = list(filter(lambda x: x in reconstructed_g, targets))

            experiment_dict = dict()
            if sources and targets:
                experiment_dict[','.join(map(lambda x: str(x), sources))] = dict(
                    map(lambda target: (str(target), 1), targets))
                with open(KEGG_GENERATED_FOLDER / kegg_file_path.stem / 'experiment.json', 'w') as f:
                    json.dump(experiment_dict, f, indent=2)

            reconstructed_experiment_dict = dict()
            if reconstructed_sources and reconstructed_targets:
                reconstructed_experiment_dict[','.join(map(lambda x: str(x), reconstructed_sources))] = dict(
                    map(lambda target: (str(target), 1), reconstructed_targets))

            all_kegg_annotations.extend(true_annotations)
            all_experiments.update(experiment_dict)
            reconstructed_experiments.update(reconstructed_experiment_dict)

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

    all_sources = []
    all_targets = []
    for sources_str in all_experiments:
        sources = sources_str.split(',')
        all_sources.extend(sources)
        all_targets.extend(list(all_experiments[sources_str].keys()))
    all_sources = list(set(all_sources).difference(all_targets))
    all_targets = list(set(all_targets).difference(all_sources))
    with open(r'C:\git\BNet21-D2D\sources\yeast\generated\KEGG\all_sources.json', 'w') as f:
        f.write(json.dumps(' '.join(all_sources)))
    with open(r'C:\git\BNet21-D2D\sources\yeast\generated\KEGG\all_targets.json', 'w') as f:
        f.write(json.dumps(' '.join(all_targets)))

    reconstructed_annotations = list(filter(lambda x: x[0] in reconstructed_g and x[1] in reconstructed_g, all_kegg_annotations))
    with open(KEGG_GENERATED_FOLDER / 'reconstructed_kegg_annotations.json', 'w') as f:
        json.dump(list(reconstructed_annotations), f)
    with open(KEGG_GENERATED_FOLDER / 'reconstructed_experiments.json', 'w') as f:
        json.dump(reconstructed_experiments, f, indent=2)

    reconstructed_g.add_edges_from(reconstructed_annotations)
    nx.write_gpickle(reconstructed_g, KEGG_GENERATED_FOLDER / 'reconstructed_net_with_test_edges.gpickle')
