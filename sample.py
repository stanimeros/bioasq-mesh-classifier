"""
Pre-sample a fixed number of articles from the full BioASQ dataset using
reservoir sampling and save them as a compact JSON file.

If --data is already a compact JSON from this script (articles use key \"text\"),
loads it once and shuffles down to max_articles — no second pass over allMeSH.

Optional quality filters (prefer denser supervision / longer text):
  --min_mesh_labels / --min_text_chars drop weak rows after the reservoir draw.
  --oversample_factor requests factor×max_articles from the stream first, then
  filters and trims back to max_articles (uses more RAM during sampling).

Usage:
    python sample.py --data data/allMeSH_2022.json --max_articles 10000 --out data/sample.json
    python sample.py --data data/sample.json --max_articles 1000 --out data/smoke.json
    python sample.py --data data/allMeSH_2022.json --max_articles 50000 --out data/sample.json \\
        --min_mesh_labels 3 --min_text_chars 400 --oversample_factor 2
"""

import argparse
import json
import random
import yaml

from dataset import load_bioasq_data


def apply_filters(texts, label_lists, min_mesh_labels, min_text_chars):
    if min_mesh_labels <= 0 and min_text_chars <= 0:
        return list(texts), list(label_lists)
    out_t, out_l = [], []
    for t, labels in zip(texts, label_lists):
        if min_mesh_labels > 0 and len(labels) < min_mesh_labels:
            continue
        if min_text_chars > 0 and len(t.strip()) < min_text_chars:
            continue
        out_t.append(t)
        out_l.append(labels)
    return out_t, out_l


def subsample_to_max(texts, label_lists, max_articles, seed):
    if max_articles is None or len(texts) <= max_articles:
        return texts, label_lists
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    idx = idx[:max_articles]
    return [texts[i] for i in idx], [label_lists[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to full BioASQ JSON / ZIP")
    parser.add_argument("--max_articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--config", default=None, help="YAML config to read max_articles from")
    parser.add_argument(
        "--min_mesh_labels",
        type=int,
        default=0,
        help="Keep articles with at least this many MeSH labels (0 = no filter)",
    )
    parser.add_argument(
        "--min_text_chars",
        type=int,
        default=0,
        help="Keep articles with at least this many characters in text (0 = no filter)",
    )
    parser.add_argument(
        "--oversample_factor",
        type=int,
        default=1,
        help="Reservoir size = max_articles × factor before filters/trim (1 = off; 2–5 typical with filters)",
    )
    args = parser.parse_args()

    max_articles = args.max_articles
    if max_articles is None and args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        max_articles = cfg.get("max_articles", 10000)
    elif max_articles is None:
        max_articles = 10000

    factor = max(1, min(args.oversample_factor, 10))
    load_cap = max_articles * factor

    print(
        f"Sampling up to {load_cap} articles from {args.data} (seed={args.seed}), "
        f"target after filters/trim={max_articles}..."
    )
    texts, label_lists = load_bioasq_data(args.data, max_articles=load_cap, seed=args.seed)
    n_loaded = len(texts)

    texts, label_lists = apply_filters(texts, label_lists, args.min_mesh_labels, args.min_text_chars)
    n_kept = len(texts)

    if args.min_mesh_labels > 0 or args.min_text_chars > 0:
        print(f"Quality filter: kept {n_kept} / {n_loaded} articles (min_mesh={args.min_mesh_labels}, min_chars={args.min_text_chars})")

    if n_kept < max_articles:
        print(
            f"Warning: only {n_kept} articles meet criteria (wanted {max_articles}). "
            "Relax filters, raise --oversample_factor, or lower --max_articles."
        )

    texts, label_lists = subsample_to_max(texts, label_lists, max_articles, args.seed)

    articles = [{"text": text, "meshMajor": labels} for text, labels in zip(texts, label_lists)]
    with open(args.out, "w") as f:
        json.dump({"articles": articles}, f)

    print(f"Saved {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
