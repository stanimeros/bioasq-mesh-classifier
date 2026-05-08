"""
Pre-sample a fixed number of articles from the full BioASQ dataset using
reservoir sampling and save them as a compact JSON file.

Usage:
    python sample.py --data data/allMeSH_2022.json --max_articles 10000 --out data/sample.json
"""

import argparse
import json
from dataset import load_bioasq_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to full BioASQ JSON / ZIP")
    parser.add_argument("--max_articles", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"Sampling {args.max_articles} articles from {args.data} (seed={args.seed})...")
    texts, label_lists = load_bioasq_data(args.data, max_articles=args.max_articles, seed=args.seed)

    articles = [
        {"text": text, "meshMajor": labels}
        for text, labels in zip(texts, label_lists)
    ]
    with open(args.out, "w") as f:
        json.dump({"articles": articles}, f)

    print(f"Saved {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
