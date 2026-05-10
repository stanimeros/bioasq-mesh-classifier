import argparse
import os
import pickle

import torch
import yaml
import wandb
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from dataset import load_bioasq_data, build_label_vocab, encode_labels, BioASQDataset
from evaluate import evaluate_transformer, find_best_threshold, save_results
from model import AsymmetricLoss, BioASQClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override data path")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases (e.g. smoke runs)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (e.g. smoke runs)")
    parser.add_argument("--max_length", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Override early stopping patience (0 disables)",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Cap number of articles after load (pre-sampled JSON only; faster smoke)",
    )
    parser.add_argument("--min_label_count", type=int, default=None, help="Override min MeSH label count for vocab")
    return parser.parse_args()


def apply_cfg_overrides(cfg, args):
    """Apply CLI overrides onto loaded YAML (mutates cfg)."""
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_length is not None:
        cfg["max_length"] = args.max_length
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.early_stopping_patience is not None:
        cfg["early_stopping_patience"] = args.early_stopping_patience
    if args.min_label_count is not None:
        cfg["min_label_count"] = args.min_label_count


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data:
        cfg["data"] = args.data
    apply_cfg_overrides(cfg, args)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = os.path.basename(cfg["output_dir"])
    wandb_kwargs = {"project": "bioasq-mesh-classifier", "name": run_name, "config": cfg}
    if args.no_wandb:
        wandb_kwargs["mode"] = "disabled"
    wandb.init(**wandb_kwargs)
    wandb.define_metric("*", step_metric="epoch")

    print("Loading data...")
    texts, label_lists = load_bioasq_data(cfg["data"], max_articles=args.max_articles)
    print(f"Loaded {len(texts)} articles")

    vocab = build_label_vocab(label_lists, min_count=cfg["min_label_count"])
    print(f"Label vocabulary: {len(vocab)}")

    with open(os.path.join(cfg["output_dir"], "label_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    label_vecs = encode_labels(label_lists, vocab)

    print(f"Loading tokenizer: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = BioASQDataset(texts, label_vecs, tokenizer, num_labels=len(vocab), max_length=cfg["max_length"])

    test_size = int(len(dataset) * cfg.get("test_split", 0.1))
    val_size = int(len(dataset) * cfg["val_split"])
    train_size = len(dataset) - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)

    model = BioASQClassifier(cfg["model_name"], num_labels=len(vocab), dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = AsymmetricLoss()

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.0))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_path = os.path.join(cfg["output_dir"], "best_model.pt")
    # First epoch must be able to beat this; otherwise tiny/smoke runs may never save a checkpoint
    # (e.g. val micro-F1 stays 0.0) and test eval would crash on a missing file.
    best_micro_f1 = float("-inf")
    patience = cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        micro_f1, macro_f1 = evaluate_transformer(model, val_loader, device, cfg.get("threshold", 0.5))
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "val_micro_f1": micro_f1, "val_macro_f1": macro_f1, "lr": current_lr}, step=epoch + 1)

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model (micro-F1={best_micro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"  -> Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                break

    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("Warning: no best checkpoint saved; evaluating last-epoch weights.")

    print("Tuning threshold on validation set...")
    best_threshold, val_micro, val_macro = find_best_threshold(model, val_loader, device)
    print(f"Best threshold: {best_threshold:.2f} (val micro-F1={val_micro:.4f})")

    micro_f1, macro_f1 = evaluate_transformer(model, test_loader, device, best_threshold)
    save_results(cfg["output_dir"], micro_f1, macro_f1, best_threshold)
    print(f"Test | threshold={best_threshold:.2f} | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f}")
    wandb.log({"test_micro_f1": micro_f1, "test_macro_f1": macro_f1, "best_threshold": best_threshold}, step=cfg["epochs"])
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
