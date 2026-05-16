import argparse
import os
import pickle
import numpy as np
from contextlib import nullcontext

import torch
import yaml
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from dataset import load_bioasq_data, build_label_vocab, encode_labels, BioASQDataset, stratified_split
from evaluate import evaluate_transformer, find_best_thresholds, save_results
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
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (fp16/bf16); slower but maximally stable",
    )
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
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        print(
            f"GPU: {props.name} | CUDA capability sm_{cap[0]}{cap[1]} | "
            f"VRAM ~{props.total_memory / (1024**3):.1f} GiB | torch {torch.__version__}"
        )
        torch.backends.cudnn.benchmark = True
    else:
        print(
            "Warning: training on CPU (no CUDA). "
            "On a Linux GPU box, reinstall torch with CUDA (see setup.sh) or set CUDA_VISIBLE_DEVICES."
        )

    run_name = os.path.basename(cfg["output_dir"])
    wandb_kwargs = {"project": "bioasq-mesh-classifier", "name": run_name, "config": cfg}
    if args.no_wandb:
        wandb_kwargs["mode"] = "disabled"
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = None
    scaler = None
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        print("AMP: bfloat16 (no loss scaling)")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        print("AMP: float16 + GradScaler")
    else:
        print("AMP: disabled")

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

    train_set, val_set, test_set = stratified_split(
        dataset, label_vecs, len(vocab),
        val_split=cfg["val_split"],
        test_split=cfg.get("test_split", 0.1),
    )
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)} (stratified)")

    num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    pin_memory = device.type == "cuda"
    dl_kw = {"num_workers": num_workers, "pin_memory": pin_memory}
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **dl_kw)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], **dl_kw)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], **dl_kw)

    model = BioASQClassifier(cfg["model_name"], num_labels=len(vocab), dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = AsymmetricLoss()

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.0))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_path = os.path.join(cfg["output_dir"], "best_model.pt")
    best_macro_f1 = float("-inf")
    patience = cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
        if use_amp and amp_dtype is not None
        else nullcontext()
    )

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        micro_f1, macro_f1 = evaluate_transformer(model, val_loader, device, np.full(len(vocab), cfg.get("threshold", 0.5)))
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "val_micro_f1": micro_f1, "val_macro_f1": macro_f1, "lr": current_lr}, step=epoch + 1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model (macro-F1={best_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                print(f"  -> Early stopping after {epoch+1} epochs (no improvement in macro-F1 for {patience} epochs)")
                break

    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("Warning: no best checkpoint saved; evaluating last-epoch weights.")

    print("Tuning threshold on validation set...")
    best_thresholds, val_micro, val_macro = find_best_thresholds(model, val_loader, device)
    print(f"Per-label thresholds tuned (val micro-F1={val_micro:.4f}, macro-F1={val_macro:.4f})")

    micro_f1, macro_f1 = evaluate_transformer(model, test_loader, device, best_thresholds)
    mean_threshold = float(best_thresholds.mean())
    save_results(cfg["output_dir"], micro_f1, macro_f1, mean_threshold)
    print(f"Test | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f} | mean threshold={mean_threshold:.3f}")
    wandb.log({"test_micro_f1": micro_f1, "test_macro_f1": macro_f1, "mean_threshold": mean_threshold}, step=cfg["epochs"])
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
