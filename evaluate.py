"""Step 2: automatic evaluation + blind sheets for human rating.

    python evaluate.py                    # reads results/outputs.jsonl
    python evaluate.py --outputs results/outputs_mock.jsonl

Writes
  results/auto_metrics.md            tables for the paper (automatic metrics)
  results/annotation_sheet.csv       blind, shuffled sheet -> give one copy to each rater
  results/annotation_key.csv         which system produced each item (do NOT show raters)

IMPORTANT: the "KB drift rate" uses the same knowledge base as the PragmaSense
critic, so PragmaSense is optimised for it. It is reported as a diagnostic only;
the main evidence is the blind human rating (analyze_human.py).
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from pragmasense import KnowledgeBase, PragmaticCritic
from pragmasense.metrics import chrf_pp
from pragmasense.pipeline import PragmaSense

ORDER = ["google_translate", "zero_shot", "cot", "self_refine", "kb_no_critic", "pragmasense", "human_reference"]
NAMES = {"google_translate": "Google Translate", "zero_shot": "Zero-shot LLM", "cot": "Chain-of-Thought",
         "self_refine": "Self-Refine (no KB)", "kb_no_critic": "KB prompt (no critic)",
         "pragmasense": "PragmaSense-AI", "human_reference": "Human reference"}
ARCH = ["deference", "softening", "urgency", "familiarity"]


TONE_HINTS = {
    "deference": "ယဉ်ကျေးပြီး လေးစားတဲ့ လေသံ",
    "softening": "နူးညံ့ ရင်းနှီးတဲ့ လေသံ (တောင်းဆို/သတိပေးတာကို ပျော့ပျော့ပြော)",
    "urgency": "အလျင်လို / တိုက်တွန်းတဲ့ လေသံ (ယဉ်ကျေးလွန်းစရာ မလို)",
    "familiarity": "အရမ်းရင်းနှီးတဲ့ သူငယ်ချင်းချင်း ကြမ်းကြမ်းပြောတဲ့ လေသံ",
}
GENDER_HINTS = {"male": " — ပြောသူ ယောက်ျား", "female": " — ပြောသူ မိန်းမ"}


def tone_hint(archetype, gender):
    return TONE_HINTS.get(archetype, "") + GENDER_HINTS.get(gender, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="results/outputs.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rate_systems", default="google_translate,zero_shot,pragmasense",
                    help="systems included in the human rating sheet (all systems still get automatic metrics)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    recs = [json.loads(l) for l in open(args.outputs, encoding="utf-8")]
    # add the human reference as an extra "system" (upper bound + sanity check for raters)
    seen = set()
    for r in list(recs):
        if r["id"] not in seen:
            seen.add(r["id"])
            recs.append({**{k: r[k] for k in ("id", "archetype", "context", "speaker_gender", "thai", "reference")},
                         "system": "human_reference", "translation": r["reference"]})

    kb = KnowledgeBase()
    critic = PragmaticCritic(kb)
    ps = PragmaSense(llm=None, kb=kb)
    for r in recs:
        prof, _ = ps.perceive(r["thai"], r["speaker_gender"])
        v = critic.audit(prof, r["translation"])
        r["delta"], r["kb_accept"] = v.delta, v.accepted

    by_sys = defaultdict(list)
    for r in recs:
        by_sys[r["system"]].append(r)
    systems = [s for s in ORDER if s in by_sys]

    md = ["# Automatic metrics\n",
          "KB drift = % outputs rejected by the knowledge-base critic (diagnostic; PragmaSense optimises this).\n",
          "| System | N | KB drift (%) | mean Δ | chrF++ | LLM calls / item | LLM time / item (s) |",
          "|---|---|---|---|---|---|---|"]
    for s in systems:
        rs = by_sys[s]
        drift = 100 * np.mean([not r["kb_accept"] for r in rs])
        delta = np.mean([r["delta"] for r in rs])
        chrf = chrf_pp([r["translation"] for r in rs], [r["reference"] for r in rs]) if s != "human_reference" else 100.0
        calls = np.mean([r.get("llm_calls", 0) for r in rs])
        lat = np.mean([r.get("latency", 0) for r in rs])
        md.append(f"| {NAMES[s]} | {len(rs)} | {drift:.1f} | {delta:.2f} | {chrf:.1f} | {calls:.2f} | {lat:.2f} |")

    md += ["\n## KB drift (%) by archetype\n",
           "| System | " + " | ".join(ARCH) + " |", "|---" * (len(ARCH) + 1) + "|"]
    for s in systems:
        cells = []
        for a in ARCH:
            rs = [r for r in by_sys[s] if r["archetype"] == a]
            cells.append(f"{100 * np.mean([not r['kb_accept'] for r in rs]):.0f}" if rs else "-")
        md.append(f"| {NAMES[s]} | " + " | ".join(cells) + " |")

    if "pragmasense" in by_sys:
        rs = by_sys["pragmasense"]
        st = defaultdict(int)
        for r in rs:
            st[r["status"]] += 1
        n_rej = len(rs) - st["accepted_draft"]
        md += ["\n## PragmaSense metacognitive loop\n",
               f"- drafts accepted without revision: {st['accepted_draft']}/{len(rs)}",
               f"- drafts rejected by critic: {n_rej}/{len(rs)}",
               f"  - fixed by revision: {st['fixed']}" + (f" ({100 * st['fixed'] / n_rej:.1f}% of rejected)" if n_rej else ""),
               f"  - stopped by gate (revision gave no gain): {st['stagnated']}",
               f"  - still rejected after k_max revisions: {st['unresolved']}",
               f"- mean revision rounds (rejected drafts only): "
               f"{np.mean([r['iterations'] for r in rs if r['status'] != 'accepted_draft']) if n_rej else 0:.2f}"]

    Path("results").mkdir(exist_ok=True)
    Path("results/auto_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---------------- blind annotation sheet (identical outputs rated once)
    rate_systems = set(x.strip() for x in args.rate_systems.split(","))
    items = {}
    for r in recs:
        if r["system"] not in rate_systems:
            continue  # only selected systems are rated (keeps the rating job small)
        key = (r["id"], r["translation"])
        items.setdefault(key, {"r": r, "systems": []})["systems"].append(r["system"])
    keys = list(items)
    notes = {}
    if Path("data/source_notes.csv").exists():
        notes = {r["id"]: r for r in csv.DictReader(open("data/source_notes.csv", encoding="utf-8"))}
    random.Random(args.seed).shuffle(keys)
    with open("results/annotation_sheet.csv", "w", newline="", encoding="utf-8-sig") as f, \
         open("results/annotation_key.csv", "w", newline="", encoding="utf-8") as g:
        ws, wk = csv.writer(f), csv.writer(g)
        # Raters are native Myanmar speakers who may not read Thai, so each item shows the
        # meaning in English and the intended tone. The human reference is NOT shown, so
        # raters are not anchored to one particular wording.
        ws.writerow(["item", "situation", "thai", "english_meaning", "intended_tone",
                     "translation_to_rate", "pragmatic_fidelity_1to5", "comment"])
        wk.writerow(["item", "id", "archetype", "systems", "delta"])
        for i, k in enumerate(keys, 1):
            it = items[k]
            r = it["r"]
            n = notes.get(r["id"], {})
            tone = tone_hint(r["archetype"], r["speaker_gender"])
            if n.get("source_tone"):
                tone += f" ({n['source_tone']})"
            ws.writerow([i, r["context"], r["thai"], n.get("english_meaning", ""), tone,
                         r["translation"], "", ""])
            wk.writerow([i, it["r"]["id"], it["r"]["archetype"], "|".join(it["systems"]), it["r"]["delta"]])
    print(f"\nWrote results/annotation_sheet.csv ({len(keys)} unique items to rate) and results/annotation_key.csv")


if __name__ == "__main__":
    main()
