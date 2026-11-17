"""Small, dependency-free metric implementations (numpy only)."""
from collections import Counter

import numpy as np


# ------------------------------------------------------------------ chrF++
def _ngrams(seq, n):
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def chrf_pp(hyps, refs, char_order=6, word_order=2, beta=2.0):
    """Corpus-level chrF++ (Popovic, 2017): character n-grams 1..6 (spaces
    removed) plus word n-grams 1..2. Returns a score in [0, 100].
    Note: Myanmar is usually written without spaces between words, so the
    word n-gram part only sees space-separated phrases."""
    orders = [("c", n) for n in range(1, char_order + 1)] + [("w", n) for n in range(1, word_order + 1)]
    stats = {o: [0, 0, 0] for o in orders}  # matches, hyp total, ref total
    for h, r in zip(hyps, refs):
        hc, rc = list(h.replace(" ", "")), list(r.replace(" ", ""))
        hw, rw = h.split(), r.split()
        for kind, n in orders:
            hn = _ngrams(hc if kind == "c" else hw, n)
            rn = _ngrams(rc if kind == "c" else rw, n)
            stats[(kind, n)][0] += sum((hn & rn).values())
            stats[(kind, n)][1] += sum(hn.values())
            stats[(kind, n)][2] += sum(rn.values())
    precs, recs = [], []
    for m, ht, rt in stats.values():
        if ht > 0 and rt > 0:
            precs.append(m / ht)
            recs.append(m / rt)
    if not precs:
        return 0.0
    p, r = float(np.mean(precs)), float(np.mean(recs))
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return 100 * (1 + b2) * p * r / (b2 * p + r)


# -------------------------------------------------------- agreement / corr
def weighted_kappa(a, b, labels=(1, 2, 3, 4, 5), weights="quadratic"):
    """Cohen's weighted kappa for two raters on an ordinal scale."""
    a, b = np.asarray(a), np.asarray(b)
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    W = ((i - j) / (k - 1)) ** 2 if weights == "quadratic" else np.abs(i - j) / (k - 1)
    denom = (W * E).sum()
    return 1.0 if denom == 0 else 1 - (W * O).sum() / denom


def _rank(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):          # average ranks for ties
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
