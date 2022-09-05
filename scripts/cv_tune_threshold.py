import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


def cv_tune_threshold(model, X: pd.DataFrame, y: pd.Series, k=5) -> float:
    """
    Helper function for binary classification threshold tuning.

    @param model: Currently used model
    @param X: X is used for split over X_train/X_val
    @param y: y is used for split over y_train/y_val
    @param k: number of k-cv splits, default=5
    @return: mean of k-thresholds
    """

    result_thresholds = []
    for i in range(k):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        model.fit(X_train, y_train)

        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc = roc_auc_score(y_val, y_pred_proba)

        ax1 = fig.add_subplot(grid[0, 0])
        ax1.set_title('ROC Curve')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.plot(fpr, tpr, label="ROC AUC = " + str(round(auc, 3)))
        ax1.legend(loc=4);

        precision_, recall_, thresholds = precision_recall_curve(y_val, y_pred_proba)
        fscore_ = (2 * precision_ * recall_) / (precision_ + recall_)
        fscore_ = fscore_[~np.isnan(fscore_)]

        ix = np.argmax(fscore_)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore_[ix]))

        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.plot(recall_, precision_)
        ax2.scatter(recall_[ix], precision_[ix], marker='o', color='green', label='Optimal')

        plt.legend()
        # plt.show()
        result_thresholds.append(thresholds[ix])

    return np.mean(result_thresholds)
