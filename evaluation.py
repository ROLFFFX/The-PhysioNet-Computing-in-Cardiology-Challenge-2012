from sklearn import metrics, exceptions

import warnings
warnings.filterwarnings('ignore', category=exceptions.UndefinedMetricWarning)

from helper import *

def performance(clf, X, y_true, metric='accuracy'):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf and X.
    Input:
        clf: an instance of sklearn estimator
        X : (N,d) np.array containing features
        y_true: (N,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'precision', 'sensitivity', 'specificity', 
                'f1-score', 'auroc', and 'auprc')
    Returns:
        the performance measure as a float
    """
    # TODO: Implement this function
    # get y_pred and confusion matrix
    y_pred = clf.predict(X)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    
    if metric == 'precision':
        return metrics.precision_score(y_true, y_pred, zero_division = 1)
    elif metric == 'sensitivity':
        return metrics.recall_score(y_true, y_pred, zero_division = 1)
    elif metric == 'specificity':
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    elif metric == 'f1_score':
        return metrics.f1_score(y_true, y_pred, zero_division = 1)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, clf.predict_proba(X)[:, 1])
    elif metric == 'auprc':
        return metrics.average_precision_score(y_true, clf.decision_function(X))
    else:
        # accuracy by default
        return metrics.accuracy_score(y_true, y_pred)