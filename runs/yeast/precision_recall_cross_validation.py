import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from d2d.d2d import get_training_features_and_scores
from d2d.propagation import PROPAGATE_METHODS
from runs.yeast.data import read_data
from runs.features import get_features
from runs.yeast.consts import YEAST_RESULTS_DIR, CROSS_VALIDATION_PRECISION_RECALL_PATH


PRIOR_WEIGHT_FUNCS = {
    'unweighted': lambda x: 1,
    'abs': abs,
    'exp abs': lambda x: 2 ** abs(x)
}


def precision_recall_cross_validation(net_type, undirected):
    network, true_annotations, experiments, edges_to_direct = read_data(net_type=net_type, undirected=undirected)

    for propagation_method in PROPAGATE_METHODS:
        fig, ax = plt.subplots()
        for prior_weight_option in PRIOR_WEIGHT_FUNCS:
            feature_cols, reverse_cols = get_features(network, experiments, edges_to_direct=edges_to_direct,
                                                      output_directory=YEAST_RESULTS_DIR, method=propagation_method,
                                                      prior_weight_func=PRIOR_WEIGHT_FUNCS[prior_weight_option],
                                                      force=True, save=False)
            logistic_regression_classifier = LogisticRegression(solver="liblinear", penalty="l1", C=0.001)

            training_feature_columns, training_reverse_columns, training_feature_scores, training_reverse_scores = \
                get_training_features_and_scores(feature_cols, reverse_cols, true_annotations)

            training_columns = pd.concat([training_feature_columns, training_reverse_columns])
            training_scores = np.append(training_feature_scores, training_reverse_scores)

            cv = StratifiedKFold(n_splits=10)

            precision_values = []
            aucs = []
            mean_recall = np.linspace(0, 1, 100)

            for i, (train, test) in enumerate(cv.split(training_columns, training_scores)):
                logistic_regression_classifier.fit(training_columns.iloc[train], training_scores[train])
                viz = plot_precision_recall_curve(logistic_regression_classifier, training_columns.iloc[test], training_scores[test])
                interp_tpr = np.interp(mean_recall, viz.recall, viz.precision)
                interp_tpr[0] = 0.0
                precision_values.append(interp_tpr)
                aucs.append(auc(viz.recall, viz.precision))

            mean_precision = np.mean(precision_values, axis=0)
            mean_precision[-1] = 1.0
            mean_auc = auc(mean_recall, mean_precision)
            std_auc = np.std(aucs)
            plt.figure(fig)
            ax.plot(mean_recall, mean_precision,
                    label=r'%s - Mean precision-recall (AUC = %0.2f $\pm$ %0.2f)' % (prior_weight_option, mean_auc, std_auc),
                    lw=2, alpha=.8)
        plt.figure(fig)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot([0, 1], [0.5, 0.5], '--', color=(0.8, 0.8, 0.8), label='random')
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=f"Precision-Recall curve - {propagation_method} diffusion")
        ax.legend(loc="lower right")

        plt.savefig(str(CROSS_VALIDATION_PRECISION_RECALL_PATH).format(propagation_method))


# def get_recall_precision(feature_columns, reverse_columns, directed_interactions):
#     directed_feature_columns, directed_reverse_columns, directed_feature_scores, directed_reverse_scores = \
#         get_training_features_and_scores(feature_columns, reverse_columns, directed_interactions)
#
#     cv = StratifiedKFold(n_splits=12)
#
#     all_proba = []
#     all_ytest = []
#     all_names = []
#
#     sum_coef_zero = 0
#
#     # Choose folds so tonot mix the true instances with the same interactions' false
#     folds = []
#     import random
#     random.shuffle(directed_interactions)
#     directed_interactions_columns = pandas.DataFrame(index=directed_interactions)
#     for i, (train_index, test_index) in enumerate(cv.split(directed_interactions, np.ones(len(directed_interactions)))):
#         training_directed_interactions = directed_interactions_columns.iloc[train_index].index
#         test_directed_interactions = directed_interactions_columns.iloc[test_index].index
#         # # Choose the corresponding false (this is done to prevent information leak caused by having one direction chosen
#         # # once in the test set and the opposite direction chosen for the training)
#         # EdgeTrue = (pred_ID2.iloc[train_index].astype(str) + "." + pred_ID1.iloc[train_index].astype(str))
#         # IdxFalse = pred_name[pred_name.isin(EdgeTrue)].index
#         # IdxFullTrain = np.append(IdxFalse.values, train_index)
#         #
#         # # The rest are the training
#         # IdxFullTest = pred_name[~pred_name.index.isin(IdxFullTrain)].index
#         folds.append((training_directed_interactions, test_directed_interactions))
#
#     for train_index, test_index in folds:
#         xtrain, xtest = directed_feature_columns.loc[train_index], feature_columns.loc[test_index]
#         # ytrain, ytest = pred_true.loc[train_index], pred_true.loc[test_index]
#         # # name_test = pred_name.loc[test_index]
#         # xtrain = xtrain.rank(axis='columns')
#         # xtest = xtest.rank(axis='columns')
#
#         training_feature_columns, training_reverse_columns, training_feature_scores, training_reverse_scores = \
#             get_training_features_and_scores(directed_feature_columns, directed_reverse_columns, directed_interactions)
#
#         training_columns = pandas.concat([training_feature_columns, training_reverse_columns])
#         training_scores = np.append(training_feature_scores, training_reverse_scores)
#
#         clf = LogisticRegression(solver="liblinear", penalty="l1", C=0.007)
#         score_network(directed_interactions_columns, reverse_columns, directed_interactions, classifier):
#         test_prob = clf.fit(xtrain, ytrain).predict_proba(xtest)
#
#         all_proba.extend(test_prob)
#         all_ytest.extend(ytest)
#         # all_names.extend(name_test)
#
#     # roc_curve
#     precision, recall, thresholds = precision_recall_curve(np.array(all_ytest), np.array(all_proba)[:, 1], pos_label=2)
#     mean_auc = auc(recall, precision)
#     rows, columns = feature_columns.shape
#
#     oriented_network = pandas.concat([pandas.DataFrame({'edge': all_names}), pandas.DataFrame({'edge': all_ytest}),
#                                       pandas.DataFrame({'pval': np.array(all_proba)[:, 1]})], axis=1)
#     oriented_network.columns = ['edge', 'true', 'pval']
#
#     return recall, precision, mean_auc, oriented_network, np.array(all_proba)[:, 1]


if __name__ == '__main__':
    precision_recall_cross_validation(net_type='biogrid', undirected=True)
