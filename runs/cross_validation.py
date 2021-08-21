import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from d2d.d2d import read_data
from runs.features import get_features
from runs.consts import FEATURE_COLS_PATH, REVERSE_COLS_PATH, CROSS_VALIDATION_ROC_PATH


def cross_validation(feature_columns, reverse_columns, directed_interactions, classifier):
    opposite_directed_interactions = list(map(lambda e: (e[1], e[0]), directed_interactions))

    true_in_feature = set(directed_interactions).intersection(feature_columns.index)
    false_in_feature = set(opposite_directed_interactions).intersection(feature_columns.index)

    training_columns = pandas.concat([feature_columns.loc[true_in_feature, :], feature_columns.loc[false_in_feature, :],
                                      reverse_columns.loc[false_in_feature, :],
                                      reverse_columns.loc[true_in_feature, :]])

    true_feature_labels = np.ones(len(true_in_feature))
    false_feature_labels = np.zeros(len(false_in_feature))
    true_reverse_labels = np.ones(len(false_in_feature))
    false_reverse_labels = np.zeros(len(true_in_feature))

    feature_labels = np.append(true_feature_labels, false_feature_labels)
    reverse_labels = np.append(true_reverse_labels, false_reverse_labels)
    training_scores = np.append(feature_labels, reverse_labels)

    cv = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(training_columns, training_scores)):
        classifier.fit(training_columns.iloc[train], training_scores[train])
        viz = plot_roc_curve(classifier, training_columns.iloc[test], training_scores[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC curve")
    ax.legend(loc="lower right")
    plt.savefig(CROSS_VALIDATION_ROC_PATH)


if __name__ == '__main__':
    network, true_annotations, experiments = read_data()
    feature_cols, reverse_cols = get_features(network, experiments,
                                              FEATURE_COLS_PATH, REVERSE_COLS_PATH)
    logistic_regression_classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001)
    cross_validation(feature_cols, reverse_cols, true_annotations, logistic_regression_classifier)
