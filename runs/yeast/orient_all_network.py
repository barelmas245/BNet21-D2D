import networkx as nx
from sklearn.linear_model import LogisticRegression

from d2d.d2d import score_network, orient_edges, ORIENTATION_EPSILON
from runs.yeast.data import read_data
from runs.features import get_features
from runs.yeast.consts import YEAST_RESULTS_DIR, ANNOTATED_NETWORK_PATH


if __name__ == '__main__':
    import time
    network, true_annotations, experiments, edges_to_direct = read_data(net_type='biogrid', undirected=True)
    start_time = time.time()
    feature_columns, reverse_columns = get_features(network, experiments, edges_to_direct=edges_to_direct,
                                                    output_directory=YEAST_RESULTS_DIR, force=True, save=False)
    print(f"FEATURES CALCULATIONS: {time.time() - start_time}")
    classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001)
    scores = score_network(feature_columns, reverse_columns, true_annotations, classifier)
    annotated_network, annotated_edges = orient_edges(scores, orientation_epsilon=ORIENTATION_EPSILON)
    end_time = time.time()
    print(f"TOTAL TIME: {end_time - start_time}")
    nx.write_gpickle(annotated_network, ANNOTATED_NETWORK_PATH)
