import argparse
import json
import pickle
import os

import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import load_bioasq_data
from model import BioASQClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to BioASQ JSON test file")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model_name", default="dmis-lab/biobert-base-cased-v1.2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.output_dir, "label_vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    id_to_label = {i: label for label, i in vocab.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BioASQClassifier(args.model_name, num_labels=len(vocab))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    model.to(device)
    model.eval()

    texts, _ = load_bioasq_data(args.data)

    results = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch_texts = texts[i:i + args.batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

        for prob_vec in probs:
            predicted = [id_to_label[j] for j, p in enumerate(prob_vec) if p >= args.threshold]
            results.append(predicted)

    out_path = os.path.join(args.output_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
