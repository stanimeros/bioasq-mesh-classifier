import json
import torch
from torch.utils.data import Dataset


def load_bioasq_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data["articles"]
    texts, label_lists = [], []
    for article in articles:
        title = article.get("title", "")
        abstract = article.get("abstractText", "")
        text = title + " [SEP] " + abstract
        # meshMajor is a list of plain strings e.g. ["Humans", "Animals", ...]
        mesh_labels = article.get("meshMajor", [])
        if mesh_labels and (title or abstract):
            texts.append(text)
            label_lists.append(mesh_labels)
    return texts, label_lists


def build_label_vocab(label_lists, min_count=10):
    from collections import Counter
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
    """Lazy-tokenizing dataset — tokenization happens per sample to avoid OOM on large corpora."""

    def __init__(self, texts, label_vecs, tokenizer, max_length=512):
        self.texts = texts
        self.labels = label_vecs  # list of tensors
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
