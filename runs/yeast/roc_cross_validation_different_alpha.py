import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from d2d.d2d import get_training_features_and_scores
from d2d.propagation import RWR_PROPAGATION
from runs.yeast.data import read_data
from runs.features import get_features
from runs.yeast.consts import YEAST_RESULTS_DIR, CROSS_VALIDATION_ROC_PATH


PRIOR_WEIGHT_FUNCS = {
    'uniform': lambda x: 1,
    'absolute': abs,
    'exp absolute': lambda x: 2 ** abs(x)
}


def roc_cross_validation(net_type, undirected):
    network, true_annotations, experiments, edges_to_direct = read_data(net_type=net_type, undirected=undirected)

    for prior_weight_option in PRIOR_WEIGHT_FUNCS:
        fig, ax = plt.subplots()
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            feature_cols, reverse_cols = get_features(network, experiments, edges_to_direct=edges_to_direct, alpha=alpha,
                                                      output_directory=YEAST_RESULTS_DIR, method=RWR_PROPAGATION,
                                                      prior_weight_func=PRIOR_WEIGHT_FUNCS[prior_weight_option],
                                                      force=True, save=False)
            logistic_regression_classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001)

            training_feature_columns, training_reverse_columns, training_feature_scores, training_reverse_scores = \
                get_training_features_and_scores(feature_cols, reverse_cols, true_annotations)

            training_columns = pandas.concat([training_feature_columns, training_reverse_columns])
            training_scores = np.append(training_feature_scores, training_reverse_scores)

            cv = StratifiedKFold(n_splits=10)

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for i, (train, test) in enumerate(cv.split(training_columns, training_scores)):
                logistic_regression_classifier.fit(training_columns.iloc[train], training_scores[train])
                viz = plot_roc_curve(logistic_regression_classifier, training_columns.iloc[test], training_scores[test])
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.figure(fig)
            ax.plot(mean_fpr, mean_tpr,
                    label=r'alpha=%0.2f (AUC = %0.2f $\pm$ %0.2f)' % (alpha, mean_auc, std_auc),
                    lw=2, alpha=.8)
        plt.figure(fig)
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=f"Mean ROC curve - RWR diffusion, {prior_weight_option} prior")
        ax.legend(loc="lower right")

        plt.savefig(str(CROSS_VALIDATION_ROC_PATH).format(f'{prior_weight_option}_prior'))


if __name__ == '__main__':
    roc_cross_validation(net_type='anat', undirected=True)
