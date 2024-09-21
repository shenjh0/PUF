import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def bi_pr(confusion_matrix):
    precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])  # TP/(TP+FP)
    recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])  # TP/(TP+FN)
    f1_score = 2*precision*recall/(precision+recall)

    return precision, recall, f1_score

def _kappa(confusion_matrix):
    N = np.sum(confusion_matrix)
    p0 = np.trace(confusion_matrix) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / (N ** 2)
    kappa = (p0 - pc) / (1 - pc)
    return kappa

def classify_metrics(label_true, label_pred, n_class=2):
    confusion_matrix = _fast_hist(label_true, label_pred, n_class)
    p, r, f1 = bi_pr(confusion_matrix)
    kappa = _kappa(confusion_matrix)
    return p, r, f1, kappa