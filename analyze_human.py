"""Step 3: analyse the blind human ratings.

Each rater fills a copy of results/annotation_sheet.csv and saves it as
results/ratings_<name>.csv  (e.g. ratings_may.csv, ratings_rater2.csv).

    python analyze_human.py

Writes results/human_metrics.md with:
  - mean +- sd pragmatic fidelity per system (meaning is covered by chrF++)
  - "human drift rate" = % items whose mean pragmatic score is <= 2
  - pragmatic fidelity by archetype
  - quadratic-weighted Cohen's kappa between every pair of raters
  - Spearman correlation between the KB divergence (delta) and human scores
    (checks whether our self-defined metric agrees with people)
"""
import csv
import glob
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import ARCH, NAMES, ORDER
from pragmasense.metrics import spearman, weighted_kappa


def read_ratings(path):
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        p = row["pragmatic_fidelity_1to5"].strip()
        if p:
            out[int(row["item"])] = int(float(p))
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(glob.glob("results/ratings_*.csv"))
    if not files:
        print("No results/ratings_*.csv files found.")
        return
    raters = {Path(f).stem.replace("ratings_", ""): read_ratings(f) for f in files}
    key = {int(r["item"]): r for r in csv.DictReader(open("results/annotation_key.csv", encoding="utf-8"))}
    common = sorted(set.intersection(*[set(v) for v in raters.values()]) & set(key))
    print(f"raters: {list(raters)} | items rated by all: {len(common)}/{len(key)}")

    md = [f"# Human evaluation\n", f"Raters: {', '.join(raters)} | items rated by all raters: {len(common)}\n"]

    # agreement
    if len(raters) >= 2:
        md += ["## Inter-annotator agreement (quadratic weighted κ)\n", "| Raters | κ |", "|---|---|"]
        for a, b in itertools.combinations(raters, 2):
            kp = weighted_kappa([raters[a][i] for i in common], [raters[b][i] for i in common])
            md.append(f"| {a} vs {b} | {kp:.2f} |")

    # mean score per item, then spread back to every system that produced it
    per_sys = defaultdict(list)
    deltas, prag_means = [], []
    for i in common:
        p = np.mean([raters[r][i] for r in raters])
        k = key[i]
        deltas.append(float(k["delta"]))
        prag_means.append(p)
        for s in k["systems"].split("|"):
            per_sys[s].append((p, k["archetype"]))

    systems = [s for s in ORDER if s in per_sys]
    md += ["\n## Scores per system (1–5, mean ± sd)\n",
           "| System | N | Pragmatic fidelity | Human drift (% ≤ 2) |", "|---|---|---|---|"]
    for s in systems:
        p = np.array([x[0] for x in per_sys[s]])
        md.append(f"| {NAMES[s]} | {len(p)} | {p.mean():.2f} ± {p.std(ddof=1) if len(p) > 1 else 0:.2f} | "
                  f"{100 * np.mean(p <= 2):.1f} |")

    md += ["\n## Pragmatic fidelity by archetype\n",
           "| System | " + " | ".join(ARCH) + " |", "|---" * (len(ARCH) + 1) + "|"]
    for s in systems:
        cells = []
        for a in ARCH:
            xs = [x[0] for x in per_sys[s] if x[1] == a]
            cells.append(f"{np.mean(xs):.2f}" if xs else "-")
        md.append(f"| {NAMES[s]} | " + " | ".join(cells) + " |")

    rho = spearman(deltas, prag_means)
    md += ["\n## Validity of the Δ metric\n",
           f"Spearman ρ between KB divergence Δ and mean human pragmatic score: {rho:.2f} "
           f"(n = {len(common)}; negative = higher Δ goes with lower human score, as expected)"]

    Path("results/human_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
