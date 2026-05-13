"""
Pre-sample articles from the full BioASQ dataset and save as compact JSON.

Strategies:
  reservoir (default) — uniform reservoir sampling over the full stream.
  coverage            — greedy single-pass: include any article with an
                        undercovered label (< --min_per_label occurrences),
                        then fill remaining slots with reservoir sampling.
                        Guarantees every label seen in the data appears at
                        least min_per_label times (or as often as it occurs).

Usage:
    python sample.py --data data/allMeSH_2022.json --max_articles 50000 --out data/sample.json
    python sample.py --data data/allMeSH_2022.json --max_articles 50000 --out data/sample.json \\
        --strategy coverage --min_per_label 5
    python sample.py --data data/sample.json --max_articles 1000 --out data/smoke.json
"""

import argparse
import json
import random

from dataset import load_bioasq_data, stream_bioasq_articles


def reservoir_sample(data_path, max_articles, seed):
    texts, label_lists = load_bioasq_data(data_path, max_articles=max_articles, seed=seed)
    return texts, label_lists


def coverage_sample(data_path, max_articles, min_per_label, seed):
    """Single-pass coverage-first sampler.

    Greedily includes articles that cover undercovered labels, then fills
    remaining slots with uniform reservoir sampling.
    """
    rng = random.Random(seed)
    label_counts = {}
    coverage = []       # articles selected for label coverage
    fill_reservoir = [] # reservoir for fill articles (all labels already met)
    fill_seen = 0

    for text, labels in stream_bioasq_articles(data_path):
        if any(label_counts.get(l, 0) < min_per_label for l in labels):
            coverage.append((text, labels))
            for l in labels:
                label_counts[l] = label_counts.get(l, 0) + 1
        else:
            fill_seen += 1
            if len(fill_reservoir) < max_articles:
                fill_reservoir.append((text, labels))
            else:
                j = rng.randint(0, fill_seen - 1)
                if j < max_articles:
                    fill_reservoir[j] = (text, labels)

    unique_labels = len(label_counts)
    print(f"Coverage pass: {len(coverage)} articles cover {unique_labels} unique labels "
          f"(min_per_label={min_per_label})")

    if len(coverage) >= max_articles:
        rng.shuffle(coverage)
        selected = coverage[:max_articles]
        print(f"Coverage articles ({len(coverage)}) exceeded target; subsampled to {max_articles}")
    else:
        needed = max_articles - len(coverage)
        fill = fill_reservoir[:needed]
        selected = coverage + fill
        print(f"Added {len(fill)} fill articles (total {len(selected)})")

    rng.shuffle(selected)
    return [s[0] for s in selected], [s[1] for s in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to full BioASQ JSON / ZIP or pre-sampled JSON")
    parser.add_argument("--max_articles", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--strategy", choices=["reservoir", "coverage"], default="reservoir",
                        help="Sampling strategy (default: reservoir)")
    parser.add_argument("--min_per_label", type=int, default=5,
                        help="Min occurrences per label in coverage strategy (default: 5)")
    args = parser.parse_args()

    print(f"Sampling up to {args.max_articles} articles from {args.data} "
          f"(strategy={args.strategy}, seed={args.seed})...")

    if args.strategy == "coverage":
        texts, label_lists = coverage_sample(args.data, args.max_articles, args.min_per_label, args.seed)
    else:
        texts, label_lists = reservoir_sample(args.data, args.max_articles, args.seed)

    articles = [{"text": text, "meshMajor": labels} for text, labels in zip(texts, label_lists)]
    with open(args.out, "w") as f:
        json.dump({"articles": articles}, f)

    print(f"Saved {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
