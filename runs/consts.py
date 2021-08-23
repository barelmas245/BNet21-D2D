from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = MAIN_DIR / 'results'

FEATURE_COLS_PATH = RESULTS_DIR / r'features_cols.pickle'
REVERSE_COLS_PATH = RESULTS_DIR / r'reverse_cols.pickle'
CROSS_VALIDATION_ROC_PATH = RESULTS_DIR / r'cross_validation_roc.png'
ANNOTATED_NETWORK_PATH = RESULTS_DIR / r'annotated_network.gpickle'