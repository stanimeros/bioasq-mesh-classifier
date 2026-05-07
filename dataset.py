import torch
import ijson
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm


def load_bioasq_data(path, max_articles=None):
    """Streaming parser for BioASQ allMeSH JSON format.

    Uses ijson to stream articles from the top-level 'articles' array
    without loading the entire file into memory.
    """
    texts, label_lists = [], []
    with open(path, "rb") as f:
        articles_iter = ijson.items(f, "articles.item", use_float=True)
        for article in tqdm(articles_iter, desc="Loading", unit=" articles"):
            title = article.get("title", "")
            abstract = article.get("abstractText", "")
            mesh_labels = article.get("meshMajor", [])
            if mesh_labels and (title or abstract):
                texts.append(title + " [SEP] " + abstract)
                label_lists.append(mesh_labels)
            if max_articles and len(texts) >= max_articles:
                break
    return texts, label_lists


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
