"""Run all systems on the benchmark and save outputs to results/outputs.jsonl.

    python run_experiments.py                       # zero_shot, cot, pragmasense (light setup)
    python run_experiments.py --systems zero_shot,cot,kb_no_critic,self_refine,pragmasense   # + ablations
    python run_experiments.py --limit 5             # quick test on 5 rows
    python run_experiments.py --systems zero_shot,pragmasense
    python run_experiments.py --sleep 4             # pause between calls (free tier)

Finished LLM calls are cached, so if the run stops (quota / rate limit),
simply run the same command again and it continues where it stopped.
Google Translate is not called by code: paste its outputs into the
'google_translate' column of data/benchmark.csv and it is included automatically.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from pragmasense import LLMClient, PragmaSense
from pragmasense.llm import LLMError

ALL_SYSTEMS = ["zero_shot", "cot", "kb_no_critic", "self_refine", "pragmasense"]
# default = light setup; add the two ablations with --systems if you have time/quota
DEFAULT_SYSTEMS = ["zero_shot", "cot", "pragmasense"]
OUT = Path("results/outputs.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmark.csv")
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--k_max", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait after each real API call")
    ap.add_argument("--mock", action="store_true", help="offline code test only")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    llm = LLMClient(backend="mock" if args.mock else args.backend, model=args.model, sleep=args.sleep)
    ps = PragmaSense(llm, k_max=args.k_max)
    out_path = Path("results/outputs_mock.jsonl") if args.mock else OUT
    out_path.parent.mkdir(exist_ok=True)

    records = []
    print(f"Running {len(rows)} items x {systems} with {llm.backend}/{llm.model}")
    try:
        for i, r in enumerate(rows, 1):
            thai, ctx, g = r["thai"], r["context"], r["speaker_gender"]
            print(f"[{i}/{len(rows)}] {r['id']} {thai}")
            for s in systems:
                if s == "zero_shot":
                    res = ps.zero_shot(thai, ctx)
                elif s == "cot":
                    res = ps.chain_of_thought(thai, ctx)
                elif s == "kb_no_critic":
                    res = ps.kb_guided(thai, ctx, g)
                elif s == "self_refine":
                    res = ps.self_refine(thai, ctx)
                elif s == "pragmasense":
                    res = ps.translate(thai, ctx, g)
                else:
                    raise ValueError(f"unknown system {s}")
                records.append(_record(r, res))
                print(f"   {s:13s} -> {res.translation}" + (f"   [{res.status}]" if res.status else ""))
            if r.get("google_translate", "").strip():
                records.append({**_base(r), "system": "google_translate",
                                "translation": r["google_translate"].strip()})
    except LLMError as e:
        print(f"\nSTOPPED: {e}\nCompleted calls are cached - run the same command again later to continue.")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(records)} outputs to {out_path}  (new API calls this run: {llm.n_calls})")


def _base(r):
    return {"id": r["id"], "archetype": r["archetype"], "context": r["context"],
            "speaker_gender": r["speaker_gender"], "thai": r["thai"], "reference": r["myanmar_ref"]}


def _record(r, res):
    return {**_base(r), "system": res.system, "translation": res.translation,
            "llm_calls": res.llm_calls, "latency": round(res.latency, 3),
            "iterations": res.iterations, "status": res.status, "trace": res.trace}


if __name__ == "__main__":
    main()
