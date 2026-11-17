"""Sanity check of the knowledge base: run the critic on the human reference
translations. A good KB should accept (almost) all references. The pass rate
is reported in the paper as a validity check of the symbolic memory.

    python check_kb.py
"""
import csv
import sys

from pragmasense import KnowledgeBase, PragmaticCritic
from pragmasense.pipeline import PragmaSense

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

kb = KnowledgeBase()
critic = PragmaticCritic(kb)
ps = PragmaSense(llm=None, kb=kb)

rows = list(csv.DictReader(open("data/benchmark.csv", encoding="utf-8")))
ok = 0
for r in rows:
    prof, _ = ps.perceive(r["thai"], r["speaker_gender"])
    v = critic.audit(prof, r["myanmar_ref"])
    ok += v.accepted
    flag = "PASS" if v.accepted else "FAIL"
    print(f"{r['id']} {flag}  src={prof.marker_texts()} I_t={prof.level}  "
          f"ref={v.found_markers} I_g={v.generated_level}  delta={v.delta:.1f}"
          + (f"  missing={v.missing_functions}" if v.missing_functions else "")
          + (f"  {v.gender_error}" if v.gender_error else ""))
print(f"\nReference acceptance: {ok}/{len(rows)} = {ok / len(rows):.1%}")
