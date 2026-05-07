import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import yaml
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from dataset import load_bioasq_data, build_label_vocab
from evaluate import evaluate_sklearn, save_results


def tokenize(texts):
    return [simple_preprocess(t) for t in texts]


def embed(texts_tokenized, w2v_model):
    dim = w2v_model.vector_size
    vecs = []
    for tokens in texts_tokenized:
        token_vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        vecs.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(dim))
    return np.array(vecs)


def encode_labels_np(label_lists, vocab):
    Y = np.zeros((len(label_lists), len(vocab)), dtype=np.int8)
    for i, labels in enumerate(label_lists):
        for label in labels:
            if label in vocab:
                Y[i, vocab[label]] = 1
    return Y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data", default=None, help="Override data path")
    parser.add_argument("--max_articles", type=int, default=None, help="Cap articles (smoke test)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data:
        cfg["data"] = args.data
    if args.max_articles:
        cfg["max_articles"] = args.max_articles

    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("Loading data...")
    texts, label_lists = load_bioasq_data(cfg["data"], max_articles=cfg.get("max_articles"))
    print(f"Loaded {len(texts)} articles")

    vocab = build_label_vocab(label_lists, min_count=cfg["min_label_count"])
    print(f"Label vocabulary: {len(vocab)}")

    with open(os.path.join(cfg["output_dir"], "label_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    tokenized = tokenize(texts)

    print("Training Word2Vec...")
    w2v = Word2Vec(
        sentences=tokenized,
        vector_size=cfg["w2v_dim"],
        window=cfg["w2v_window"],
        min_count=cfg["w2v_min_count"],
        workers=4,
        epochs=cfg["w2v_epochs"],
    )
    w2v.save(os.path.join(cfg["output_dir"], "word2vec.model"))

    print("Embedding documents...")
    X = embed(tokenized, w2v)
    Y = encode_labels_np(label_lists, vocab)

    split = int(len(X) * (1 - cfg["val_split"]))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    print("Training MLP classifier...")
    clf = OneVsRestClassifier(
        MLPClassifier(hidden_layer_sizes=tuple(cfg["mlp_hidden_layers"]), max_iter=cfg["mlp_max_iter"]),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)

    micro_f1, macro_f1 = evaluate_sklearn(clf, X_val, Y_val)
    save_results(cfg["output_dir"], micro_f1, macro_f1)
    print(f"Word2Vec + MLP | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f}")

    with open(os.path.join(cfg["output_dir"], "mlp.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print("Done.")


if __name__ == "__main__":
    main()
