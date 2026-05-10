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


def evaluate_transformer(model, loader, device, threshold):
    probs, labels = collect_logits(model, loader, device)
    preds = (probs >= threshold).astype(int)
    return compute_f1(labels, preds)


def find_best_threshold(model, loader, device, thresholds=None):
    """Single val-set pass; sweep thresholds and return (best_threshold, micro_f1, macro_f1)."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.55, 0.05)
    probs, labels = collect_logits(model, loader, device)
    best_t, best_micro, best_macro = 0.5, 0.0, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        micro, macro = compute_f1(labels, preds)
        if micro > best_micro:
            best_t, best_micro, best_macro = t, micro, macro
    return best_t, best_micro, best_macro


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
