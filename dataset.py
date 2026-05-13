import torch
import ijson
import random
import zipfile
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm


class _Utf8CleanStream:
    """Wraps a binary file, replacing invalid UTF-8 bytes on the fly in chunks."""
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


def _open_bioasq(path):
    """Open a BioASQ JSON or ZIP file for streaming."""
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path, "r")
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        stream = _Utf8CleanStream(zf.open(name))
        stream._zf = zf
        return stream
    return _Utf8CleanStream(open(path, "rb"))


def stream_bioasq_articles(path):
    """Yield (text, labels) from raw allMeSH JSON/ZIP, skipping entries without labels or text."""
    with _open_bioasq(path) as f:
        for article in tqdm(ijson.items(f, "articles.item", use_float=True), desc="Streaming", unit=" art"):
            title = article.get("title", "")
            abstract = article.get("abstractText", "")
            labels = article.get("meshMajor", [])
            if not (labels and (title or abstract)):
                continue
            yield title + " [SEP] " + abstract, labels


def load_bioasq_data(path, max_articles=None, seed=42):
    """Streaming parser for BioASQ allMeSH JSON format.

    Handles both plain JSON and ZIP-compressed JSON (BioASQ distribution format).
    Uses ijson to stream articles from the top-level 'articles' array.

    When max_articles is set, reservoir sampling is used so that every article
    in the file has an equal probability of being included, avoiding the bias
    of simply taking the first N entries.
    """
    import json as _json

    # Fast path: pre-sampled JSON produced by sample.py (articles have "text" key).
    if not zipfile.is_zipfile(path):
        try:
            with open(path) as _f:
                head = _json.load(_f)
            first_article = head.get("articles", [{}])[0]
            if "text" in first_article:
                texts = [a["text"] for a in head["articles"]]
                labels = [a["meshMajor"] for a in head["articles"]]
                print(f"Loaded {len(texts)} pre-sampled articles from {path}")
                if max_articles is not None and len(texts) > max_articles:
                    rng = random.Random(seed)
                    idx = list(range(len(texts)))
                    rng.shuffle(idx)
                    idx = idx[:max_articles]
                    texts = [texts[i] for i in idx]
                    labels = [labels[i] for i in idx]
                    print(f"Subsampled to {len(texts)} articles (max_articles={max_articles}, seed={seed})")
                return texts, labels
        except Exception:
            pass

    rng = random.Random(seed)
    reservoir_texts, reservoir_labels = [], []
    seen = 0

    for text, labels in stream_bioasq_articles(path):
        seen += 1
        if max_articles is None:
            reservoir_texts.append(text)
            reservoir_labels.append(labels)
        elif len(reservoir_texts) < max_articles:
            reservoir_texts.append(text)
            reservoir_labels.append(labels)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_articles:
                reservoir_texts[j] = text
                reservoir_labels[j] = labels

    return reservoir_texts, reservoir_labels


def build_label_vocab(label_lists, min_count=10):
    counter = Counter(label for labels in label_lists for label in labels)
    vocab = {label: i for i, (label, count) in enumerate(counter.most_common()) if count >= min_count}
    return vocab


def encode_labels(label_lists, vocab):
    """Store labels as lists of indices (sparse) to avoid OOM with large datasets."""
    encoded = []
    for labels in label_lists:
        indices = [vocab[label] for label in labels if label in vocab]
        encoded.append(indices)
    return encoded


class BioASQDataset(Dataset):
    """Lazy-tokenizing dataset — tokenization and label encoding happen per sample to avoid OOM."""

    def __init__(self, texts, label_indices, tokenizer, num_labels, max_length=512):
        self.texts = texts
        self.label_indices = label_indices
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        vec = torch.zeros(self.num_labels)
        for i in self.label_indices[idx]:
            vec[i] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": vec,
        }
