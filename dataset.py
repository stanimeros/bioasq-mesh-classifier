import torch
import ijson
import random
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm


def load_bioasq_data(path, max_articles=None, seed=42):
    """Streaming parser for BioASQ allMeSH JSON format.

    Handles both plain JSON and ZIP-compressed JSON (BioASQ distribution format).
    Uses ijson to stream articles from the top-level 'articles' array.

    When max_articles is set, reservoir sampling is used so that every article
    in the file has an equal probability of being included, avoiding the bias
    of simply taking the first N entries.
    """
    import zipfile
    import io

    class Utf8CleanStream:
        """Wraps a binary file, replacing invalid UTF-8 bytes on the fly in chunks
        so ijson never chokes without reading the entire file into RAM first."""
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
            if hasattr(self, '_zf'):
                self._zf.close()

    def open_json(path):
        if zipfile.is_zipfile(path):
            zf = zipfile.ZipFile(path, "r")
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            stream = Utf8CleanStream(zf.open(name))
            stream._zf = zf  # keep ZipFile alive until stream is closed
            return stream
        else:
            return Utf8CleanStream(open(path, "rb"))

    # Fast path: pre-sampled JSON produced by sample.py.
    # Detected by checking the first article for a "text" key (vs "title"/"abstractText").
    import json as _json
    is_sampled = False
    if not zipfile.is_zipfile(path):
        try:
            with open(path) as _f:
                head = _json.load(_f)
            first_article = head.get("articles", [{}])[0]
            if "text" in first_article:
                is_sampled = True
                texts = [a["text"] for a in head["articles"]]
                labels = [a["meshMajor"] for a in head["articles"]]
        except Exception:
            is_sampled = False

    if is_sampled:
        print(f"Loaded {len(texts)} pre-sampled articles from {path}")
        return texts, labels

    rng = random.Random(seed)
    reservoir_texts, reservoir_labels = [], []
    seen = 0  # valid articles seen so far

    with open_json(path) as f:
        articles_iter = ijson.items(f, "articles.item", use_float=True)
        for article in tqdm(articles_iter, desc="Loading", unit=" articles"):
            title = article.get("title", "")
            abstract = article.get("abstractText", "")
            mesh_labels = article.get("meshMajor", [])
            if not (mesh_labels and (title or abstract)):
                continue

            text = title + " [SEP] " + abstract
            seen += 1

            if max_articles is None:
                reservoir_texts.append(text)
                reservoir_labels.append(mesh_labels)
            elif len(reservoir_texts) < max_articles:
                reservoir_texts.append(text)
                reservoir_labels.append(mesh_labels)
            else:
                # Reservoir sampling: replace a random slot with probability k/seen
                j = rng.randint(0, seen - 1)
                if j < max_articles:
                    reservoir_texts[j] = text
                    reservoir_labels[j] = mesh_labels

    return reservoir_texts, reservoir_labels


def build_label_vocab(label_lists, min_count=10):
    counter = Counter(label for labels in label_lists for label in labels)
    vocab = {label: i for i, (label, count) in enumerate(counter.most_common()) if count >= min_count}
    return vocab


def encode_labels(label_lists, vocab):
    n = len(vocab)
    encoded = []
    for labels in label_lists:
        vec = torch.zeros(n)
        for label in labels:
            if label in vocab:
                vec[vocab[label]] = 1.0
        encoded.append(vec)
    return encoded


class BioASQDataset(Dataset):
    """Lazy-tokenizing dataset — tokenization happens per sample to avoid OOM."""

    def __init__(self, texts, label_vecs, tokenizer, max_length=512):
        self.texts = texts
        self.labels = label_vecs
        self.tokenizer = tokenizer
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
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }
