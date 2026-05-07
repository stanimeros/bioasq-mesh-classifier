import argparse
import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from dataset import load_bioasq_data, build_label_vocab, encode_labels, BioASQDataset
from model import BioASQClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to BioASQ JSON file")
    parser.add_argument("--model_name", default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_label_count", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_articles", type=int, default=None, help="Cap articles for smoke testing")
    return parser.parse_args()


def evaluate(model, loader, device, threshold):
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
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return micro_f1, macro_f1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    texts, label_lists = load_bioasq_data(args.data, max_articles=args.max_articles)
    print(f"Loaded {len(texts)} articles")

    vocab = build_label_vocab(label_lists, min_count=args.min_label_count)
    print(f"Label vocabulary size: {len(vocab)}")

    with open(os.path.join(args.output_dir, "label_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    label_vecs = encode_labels(label_lists, vocab)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = BioASQDataset(texts, label_vecs, tokenizer, max_length=args.max_length)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    model = BioASQClassifier(args.model_name, num_labels=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_micro_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        micro_f1, macro_f1 = evaluate(model, val_loader, device, args.threshold)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | val micro-F1={micro_f1:.4f} | val macro-F1={macro_f1:.4f}")

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved best model (micro-F1={best_micro_f1:.4f})")

    # Re-evaluate best model and save results
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    micro_f1, macro_f1 = evaluate(model, val_loader, device, args.threshold)
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"micro_f1={micro_f1:.4f}\nmacro_f1={macro_f1:.4f}\n")
    print(f"Best model | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
