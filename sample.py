"""
Pre-sample a fixed number of articles from the full BioASQ dataset using
reservoir sampling and save them as a compact JSON file.

Usage:
    python sample.py --data data/allMeSH_2022.json --max_articles 10000 --out data/sample.json
"""

import argparse
import json
import yaml
from dataset import load_bioasq_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to full BioASQ JSON / ZIP")
    parser.add_argument("--max_articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--config", default=None, help="YAML config to read max_articles from")
    args = parser.parse_args()

    max_articles = args.max_articles
    if max_articles is None and args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        max_articles = cfg.get("max_articles", 10000)
    elif max_articles is None:
        max_articles = 10000

    print(f"Sampling {max_articles} articles from {args.data} (seed={args.seed})...")
    texts, label_lists = load_bioasq_data(args.data, max_articles=max_articles, seed=args.seed)

    articles = [
        {"text": text, "meshMajor": labels}
        for text, labels in zip(texts, label_lists)
    ]
    with open(args.out, "w") as f:
        json.dump({"articles": articles}, f)

    print(f"Saved {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
