import os
from pathlib import Path

RWR_PROPAGATION = "RWR"
KERNEL_PROPAGATION = "Kernel"
PROPAGATE_METHODS = [RWR_PROPAGATION, KERNEL_PROPAGATION]
PROPAGATE_ALPHA = 0.6
PROPAGATE_EPSILON = 1e-5
PROPAGATE_ITERATIONS = 100  # For RWR propagation
PROPAGATE_SMOTH = 0.3   # For kernel propagation

ORIENTATION_EPSILON = 0.01

MAIN_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = MAIN_DIR / 'sources'
RESULTS_DIR = MAIN_DIR / 'results'

ANNOTATED_NETWORK_PATH = RESULTS_DIR / r'annotated_network.gpickle'
FEATURE_COLS_PATH = RESULTS_DIR / r'features_cols.pickle'
REVERSE_COLS_PATH = RESULTS_DIR / r'reverse_cols.pickle'
CROSS_VALIDATION_ROC_PATH = RESULTS_DIR / r'cross_validation_roc.png'
