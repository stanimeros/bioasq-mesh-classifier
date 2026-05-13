#!/usr/bin/env python3
"""
Profile BioASQ data (full allMeSH / ZIP or compact sample.json) for sampling / training decisions.

Paste the full terminal output back into the chat for interpretation.

Examples:
  python analyze_data.py --data data/sample.json
  python analyze_data.py --data data/allMeSH_2022.json --reservoir 100000
  python analyze_data.py --data data/allMeSH_2022.json --stream-full
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import zipfile
from collections import Counter

import ijson
from tqdm import tqdm

from dataset import build_label_vocab, load_bioasq_data


class Utf8CleanStream:
    def __init__(self, fileobj, chunk_size=1 << 20):
        self._f = fileobj
        self._chunk_size = chunk_size
        self._buf = b""

    def read(self, size=-1):
        if size == -1:
            raw = self._buf + self._f.read()
            self._buf = b""
        else:
            while len(self._buf) < size:
                chunk = self._f.read(self._chunk_size)
                if not chunk:
                    break
                self._buf += chunk
            raw, self._buf = self._buf[:size], self._buf[size:]
        return raw.decode("utf-8", errors="replace").encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._f.close()
        if hasattr(self, "_zf"):
            self._zf.close()


def open_bioasq_stream(path: str):
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path, "r")
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        stream = Utf8CleanStream(zf.open(name))
        stream._zf = zf
        return stream
    return Utf8CleanStream(open(path, "rb"))


def is_compact_sample(path: str) -> bool:
    """Sniff compact sample.json (articles[].text) without loading the whole file."""
    if zipfile.is_zipfile(path):
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(65536)
        return b'"articles"' in chunk and b'"text"' in chunk
    except OSError:
        return False


def bucket_n_mesh(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    return "21+"


def bucket_text_len(n: int) -> str:
    if n < 200:
        return "<200"
    if n < 400:
        return "200-399"
    if n < 800:
        return "400-799"
    if n < 1500:
        return "800-1499"
    if n < 3000:
        return "1500-2999"
    return ">=3000"


def pct(vals: list[int], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] + (vals[hi] - vals[lo]) * (k - lo))


def duplicate_rate(texts: list[str]) -> tuple[int, int]:
    seen: set[str] = set()
    dup = 0
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()
        if h in seen:
            dup += 1
        else:
            seen.add(h)
    return dup, len(texts)


def stream_full_scan(path: str, reservoir_k: int, seed: int) -> dict:
    """Single pass: global mesh counts + histograms; reservoir for length / n_mesh percentiles."""
    rng = random.Random(seed)
    seen_valid = 0
    skipped = 0
    n_mesh_hist: Counter[str] = Counter()
    tlen_hist: Counter[str] = Counter()
    mesh_freq: Counter[str] = Counter()
    mesh_assignments = 0
    res_n: list[int] = []
    res_t: list[int] = []

    with open_bioasq_stream(path) as f:
        it = ijson.items(f, "articles.item", use_float=True)
        for article in tqdm(it, desc="Stream (full)", unit=" art"):
            if "text" in article:
                text = article.get("text") or ""
                mesh_labels = article.get("meshMajor") or []
            else:
                title = article.get("title", "") or ""
                abstract = article.get("abstractText", "") or ""
                mesh_labels = article.get("meshMajor") or []
                if not (mesh_labels and (title or abstract)):
                    skipped += 1
                    continue
                text = title + " [SEP] " + abstract

            if not mesh_labels:
                skipped += 1
                continue

            seen_valid += 1
            n_m = len(mesh_labels)
            t_len = len(text)
            mesh_assignments += n_m
            n_mesh_hist[bucket_n_mesh(n_m)] += 1
            tlen_hist[bucket_text_len(t_len)] += 1
            for m in mesh_labels:
                mesh_freq[m] += 1

            if len(res_n) < reservoir_k:
                res_n.append(n_m)
                res_t.append(t_len)
            else:
                j = rng.randint(0, seen_valid - 1)
                if j < reservoir_k:
                    res_n[j] = n_m
                    res_t[j] = t_len

    return {
        "seen_valid": seen_valid,
        "skipped": skipped,
        "mesh_assignments": mesh_assignments,
        "unique_mesh": len(mesh_freq),
        "mesh_freq": mesh_freq,
        "n_mesh_hist": n_mesh_hist,
        "tlen_hist": tlen_hist,
        "res_n": res_n,
        "res_t": res_t,
    }


def vocab_sizes(label_lists: list, counts: list[int]) -> list[tuple[int, int]]:
    out = []
    for mc in counts:
        v = build_label_vocab(label_lists, min_count=mc)
        out.append((mc, len(v)))
    return out


def report_block(title: str, lines: list[str]) -> str:
    w = max(len(title) + 4, 72)
    sep = "=" * w
    return sep + "\n" + title + "\n" + sep + "\n" + "\n".join(lines) + "\n"


def analyze_corpus(texts: list[str], label_lists: list[list], *, source_note: str) -> str:
    n = len(texts)
    if n == 0:
        return report_block("CORPUS SLICE REPORT", [source_note, "", "No articles in this slice."])
    n_mesh = [len(l) for l in label_lists]
    tlen = [len(t) for t in texts]

    dup, _ = duplicate_rate(texts)
    dup_rate = dup / n if n else 0.0

    lines = [
        f"Source: {source_note}",
        f"Articles: {n}",
        f"Duplicate texts (SHA256 of full UTF-8): {dup} ({100 * dup_rate:.2f}%)",
        "",
        "MeSH labels per article:",
        f"  min={min(n_mesh)} max={max(n_mesh)} mean={sum(n_mesh)/n:.2f}",
        f"  p50={pct(n_mesh, 50):.1f} p90={pct(n_mesh, 90):.1f} p95={pct(n_mesh, 95):.1f}",
        f"  share with >=3 labels: {sum(1 for x in n_mesh if x >= 3)/n*100:.2f}%",
        f"  share with >=5 labels: {sum(1 for x in n_mesh if x >= 5)/n*100:.2f}%",
        f"  share with >=10 labels: {sum(1 for x in n_mesh if x >= 10)/n*100:.2f}%",
        "",
        "Text length (characters, title [SEP] abstract or compact text):",
        f"  min={min(tlen)} max={max(tlen)} mean={sum(tlen)/n:.1f}",
        f"  p50={pct(tlen, 50):.0f} p90={pct(tlen, 90):.0f} p95={pct(tlen, 95):.0f}",
        "",
        "Label vocabulary size vs min_label_count (same rule as train.py):",
    ]
    for mc, vs in vocab_sizes(label_lists, [5, 10, 20, 50, 100, 200]):
        lines.append(f"  min_count={mc:>3}  ->  vocab_size={vs}")

    total_assignments = sum(n_mesh)
    lines.extend(
        [
            "",
            f"Total positive label assignments in this slice: {total_assignments}",
            f"Avg labels per article (same as mean above): {total_assignments/n:.2f}",
        ]
    )

    # Top labels in this slice only
    c = Counter()
    for labs in label_lists:
        for x in labs:
            c[x] += 1
    top = c.most_common(25)
    lines.append("")
    lines.append("Top 25 MeSH strings in this slice (frequency):")
    for lab, cnt in top:
        lines.append(f"  {cnt:>7}  {lab}")

    hapax = sum(1 for _, cnt in c.items() if cnt == 1)
    lines.extend(
        [
            "",
            f"Unique MeSH strings in slice: {len(c)}",
            f"Hapax (count==1) in slice: {hapax} ({100*hapax/max(len(c),1):.1f}% of unique)",
        ]
    )

    # Suggested sample.py knobs (heuristic)
    p50m = pct(n_mesh, 50)
    p50t = pct(tlen, 50)
    lines.extend(
        [
            "",
            "--- Heuristic create_sample.sh / sample.py hints (tune to taste) ---",
            f"# If you want denser supervision than the median (~{p50m:.0f} labels/article), try e.g.:",
            f"#   MIN_MESH_LABELS=4 MIN_TEXT_CHARS=400 OVERSAMPLE_FACTOR=3 MAX_ARTICLES=25000",
            f"# Median text length here ~{p50t:.0f} chars — pick MIN_TEXT_CHARS below most 'weak' articles you want to drop.",
        ]
    )

    return report_block("CORPUS SLICE REPORT", lines)


def report_stream_full(stats: dict, path: str) -> str:
    mesh_freq: Counter = stats["mesh_freq"]
    top = mesh_freq.most_common(25)
    res_n = stats["res_n"]
    res_t = stats["res_t"]
    n = stats["seen_valid"]

    lines = [
        f"File: {path}",
        f"Mode: single full scan (ijson), no text stored",
        f"Valid articles: {stats['seen_valid']}",
        f"Skipped (no mesh or no text): {stats['skipped']}",
        f"Unique MeSH strings observed: {stats['unique_mesh']}",
        f"Total mesh assignments: {stats['mesh_assignments']}",
        "",
        "MeSH labels per article (bucket counts):",
    ]
    for k in ["1", "2", "3-5", "6-10", "11-20", "21+"]:
        lines.append(f"  {k:>5}: {stats['n_mesh_hist'].get(k, 0)}")
    lines.append("")
    lines.append("Text length (bucket counts):")
    for k in ["<200", "200-399", "400-799", "800-1499", "1500-2999", ">=3000"]:
        lines.append(f"  {k:>10}: {stats['tlen_hist'].get(k, 0)}")

    lines.extend(
        [
            "",
            f"Reservoir ~{len(res_n)} articles: labels/article p50={pct(res_n, 50):.1f} p90={pct(res_n, 90):.1f}",
            f"Reservoir: text chars p50={pct(res_t, 50):.0f} p90={pct(res_t, 90):.0f}",
            "",
            "Top 25 MeSH strings in full scan (frequency):",
        ]
    )
    for lab, cnt in top:
        lines.append(f"  {cnt:>9}  {lab}")

    hapax = sum(1 for _, cnt in mesh_freq.items() if cnt == 1)
    lines.extend(
        [
            "",
            f"Hapax labels in full scan: {hapax} ({100*hapax/max(len(mesh_freq),1):.1f}% of unique)",
            "",
            "Note: full-scan label counts are global for this file; reservoir stats estimate article-level spread.",
        ]
    )

    return report_block("FULL-STREAM REPORT", lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze BioASQ JSON for sampling.")
    ap.add_argument("--data", required=True, help="Path to allMeSH JSON/ZIP or compact sample.json")
    ap.add_argument(
        "--stream-full",
        action="store_true",
        help="One streaming pass over entire file (slow on huge corpus; best global label stats)",
    )
    ap.add_argument(
        "--reservoir",
        type=int,
        default=100_000,
        help="For non-stream mode on raw BioASQ: reservoir size passed to load_bioasq_data (default 100000). Ignored for --stream-full.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stream-reservoir-k",
        type=int,
        default=50_000,
        help="In --stream-full, reservoir size for percentile estimates (default 50000)",
    )
    args = ap.parse_args()
    path = args.data

    if not os.path.isfile(path):
        raise SystemExit(f"Not a file: {path}")

    size_mb = os.path.getsize(path) / (1024**2)
    print(f"File size: {size_mb:.1f} MiB\n")

    if args.stream_full:
        stats = stream_full_scan(path, args.stream_reservoir_k, args.seed)
        print(report_stream_full(stats, path))
        return

    compact = is_compact_sample(path)
    if compact:
        texts, label_lists = load_bioasq_data(path, max_articles=None, seed=args.seed)
        note = f"compact JSON (all {len(texts)} articles loaded)"
    else:
        texts, label_lists = load_bioasq_data(path, max_articles=args.reservoir, seed=args.seed)
        note = f"raw BioASQ reservoir N={args.reservoir} (same mechanism as sample.py before filters)"

    print(analyze_corpus(texts, label_lists, source_note=note))


if __name__ == "__main__":
    main()
