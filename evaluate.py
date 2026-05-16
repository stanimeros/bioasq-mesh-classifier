import numpy as np
import torch
from sklearn.metrics import f1_score


def collect_logits(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels)
    return np.vstack(all_probs), np.vstack(all_labels)


def evaluate_transformer(model, loader, device, thresholds):
    probs, labels = collect_logits(model, loader, device)
    thresholds = np.asarray(thresholds)
    preds = (probs >= thresholds).astype(int)
    return compute_f1(labels, preds)


def find_best_thresholds(model, loader, device, candidates=None):
    """Per-label threshold tuning optimized for macro-F1.

    Returns (thresholds_array, micro_f1, macro_f1) where thresholds_array has
    one threshold per label.
    """
    if candidates is None:
        candidates = np.arange(0.1, 0.60, 0.05)
    probs, labels = collect_logits(model, loader, device)

    num_labels = probs.shape[1]
    best_thresholds = np.full(num_labels, 0.5)

    for j in range(num_labels):
        p_j = probs[:, j]
        y_j = labels[:, j]
        if y_j.sum() == 0:
            continue
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            pred_j = (p_j >= t).astype(int)
            f1 = f1_score(y_j, pred_j, average="binary", zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        best_thresholds[j] = best_t

    preds = (probs >= best_thresholds).astype(int)
    micro, macro = compute_f1(labels, preds)
    return best_thresholds, micro, macro


def evaluate_sklearn(clf, X_val, Y_val):
    Y_pred = clf.predict(X_val)
    return compute_f1(Y_val, Y_pred)


def compute_f1(Y_true, Y_pred):
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    return micro, macro


def save_results(output_dir, micro_f1, macro_f1, threshold=0.5):
    import os
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"micro_f1={micro_f1:.4f}\nmacro_f1={macro_f1:.4f}\nthreshold={threshold:.2f}\n")
