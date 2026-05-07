"""
Word embeddings baseline: Word2Vec (trained on corpus) + average pooling + MLP.
Satisfies the 'word embeddings' requirement of the assignment.
"""
import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

from dataset import load_bioasq_data, build_label_vocab


def tokenize(texts):
    return [simple_preprocess(t) for t in texts]


def embed(texts_tokenized, w2v_model):
    vecs = []
    dim = w2v_model.vector_size
    for tokens in texts_tokenized:
        token_vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        vecs.append(np.mean(token_vecs, axis=0) if token_vecs else np.zeros(dim))
    return np.array(vecs)


def encode_labels_np(label_lists, vocab):
    n = len(vocab)
    Y = np.zeros((len(label_lists), n), dtype=np.int8)
    for i, labels in enumerate(label_lists):
        for label in labels:
            if label in vocab:
                Y[i, vocab[label]] = 1
    return Y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_articles", type=int, default=None)
    parser.add_argument("--min_label_count", type=int, default=10)
    parser.add_argument("--w2v_dim", type=int, default=200)
    parser.add_argument("--w2v_epochs", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--output_dir", default="output/baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    texts, label_lists = load_bioasq_data(args.data, max_articles=args.max_articles)
    print(f"Loaded {len(texts)} articles")

    vocab = build_label_vocab(label_lists, min_count=args.min_label_count)
    print(f"Label vocabulary: {len(vocab)}")

    with open(os.path.join(args.output_dir, "label_vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    print("Tokenizing...")
    tokenized = tokenize(texts)

    print("Training Word2Vec...")
    w2v = Word2Vec(sentences=tokenized, vector_size=args.w2v_dim, window=5,
                   min_count=2, workers=4, epochs=args.w2v_epochs)
    w2v.save(os.path.join(args.output_dir, "word2vec.model"))

    print("Embedding documents...")
    X = embed(tokenized, w2v)
    Y = encode_labels_np(label_lists, vocab)

    split = int(len(X) * (1 - args.val_split))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    print("Training MLP classifier...")
    clf = OneVsRestClassifier(
        MLPClassifier(hidden_layer_sizes=(512,), max_iter=20, verbose=False),
        n_jobs=-1
    )
    clf.fit(X_train, Y_train)

    print("Evaluating...")
    Y_pred = clf.predict(X_val)
    micro_f1 = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
    print(f"Word2Vec + MLP | micro-F1={micro_f1:.4f} | macro-F1={macro_f1:.4f}")

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"micro_f1={micro_f1:.4f}\nmacro_f1={macro_f1:.4f}\n")

    with open(os.path.join(args.output_dir, "mlp.pkl"), "wb") as f:
        pickle.dump(clf, f)

    print("Done.")


if __name__ == "__main__":
    main()
