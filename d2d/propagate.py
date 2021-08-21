import math
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import norm

RWR_PROPAGATION = "RWR"
KERNEL_PROPAGATION = "Kernel"
PROPAGATE_METHODS = [RWR_PROPAGATION, KERNEL_PROPAGATION]
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100  # For RWR propagation
PROPAGATE_SMOOTH = 0.3   # For kernel propagation


def generate_similarity_matrix(graph, alpha, method):
    if method == RWR_PROPAGATION:
        matrix = nx.to_scipy_sparse_matrix(graph, graph.nodes)
        norm_matrix = sparse.diags(1 / np.sqrt(matrix.sum(0).A1))
        matrix = norm_matrix * matrix * norm_matrix
        return alpha * matrix
    if method == KERNEL_PROPAGATION:
        matrix = nx.normalized_laplacian_matrix(graph)
        return alpha * matrix
    else:
        raise ValueError("Unsupported propagation method")


def propagate(seeds_dict, matrix, gene_to_index,
              alpha, epsilon, method,
              num_iterations=PROPAGATE_ITERATIONS, smooth=PROPAGATE_SMOOTH, prior_weight_func=abs):
    num_genes = matrix.shape[0]
    curr_scores = np.zeros(num_genes)

    # Set the prior scores
    for gene in seeds_dict:
        assert gene in gene_to_index, f"Not found gene {gene} in network!"
        curr_scores[gene_to_index[gene]] = prior_weight_func(seeds_dict[gene])
    # Normalize the prior scores
    curr_scores = curr_scores / sum(curr_scores)

    if method == RWR_PROPAGATION:
        prior_vec = (1 - alpha) * curr_scores
        for _ in range(num_iterations):
            new_scores = curr_scores.copy()
            curr_scores = matrix.dot(new_scores) + prior_vec

            if math.sqrt(norm(new_scores - curr_scores)) < epsilon:
                break
        return curr_scores
    if method == KERNEL_PROPAGATION:
        from scipy.sparse.linalg import expm_multiply
        return expm_multiply(-matrix, curr_scores, start=0, stop=smooth, endpoint=True)[-1]
    else:
        raise ValueError("Unsupported propagation method")
