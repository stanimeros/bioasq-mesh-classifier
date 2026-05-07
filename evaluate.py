import numpy as np
import torch
from sklearn.metrics import f1_score


def evaluate_transformer(model, loader, device, threshold):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits).cpu().numpy() >= threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels)
    return compute_f1(np.vstack(all_labels), np.vstack(all_preds))


def evaluate_sklearn(clf, X_val, Y_val):
    Y_pred = clf.predict(X_val)
    return compute_f1(Y_val, Y_pred)


def compute_f1(Y_true, Y_pred):
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    return micro, macro


def save_results(output_dir, micro_f1, macro_f1):
    import os
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"micro_f1={micro_f1:.4f}\nmacro_f1={macro_f1:.4f}\n")
