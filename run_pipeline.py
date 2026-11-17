"""Demo: translate one Thai utterance and print the full cognitive trace.

    python run_pipeline.py --text "ไปทานข้าวด้วยกันไหมครับ"
    python run_pipeline.py --text "หิวว่ะ" --context "two close male friends"
    python run_pipeline.py --mock      # offline code test, NOT a real translation
"""
import argparse
import sys

from pragmasense import LLMClient, PragmaSense

LINE = "=" * 64


def show(res):
    for step in res.trace:
        if step["stage"] == "perception":
            print("\n[Tier 1] Sensory perception")
            print(f"  particles found     : {', '.join(step['particles']) or '(none)'}")
            print(f"  propositional part  : {step['propositional']}")
            print(f"  pragmatic functions : {', '.join(step['functions']) or '(plain)'}")
            print(f"  speaker gender      : {step['gender']}")
            print(f"  target level I_t    : {step['target_level']:.1f} / 4")
        else:
            title = "[Tier 2] System-1 draft" if step["stage"] == "draft" else f"[Tier 3 -> 2] {step['stage']}"
            print(f"\n{title}")
            print(f"  hypothesis          : {step['hypothesis']}")
            print(f"  Myanmar markers     : {', '.join(step['markers']) or '(none)'}")
            print(f"  delta = |{step['target_level']:.1f} - {step['generated_level']:.1f}| = {step['delta']:.2f}")
            if step["missing"]:
                print(f"  missing functions   : {', '.join(step['missing'])}")
            if step["gender_error"]:
                print(f"  gender error        : {step['gender_error']}")
            if "gate" in step:
                print(f"  convergence gate    : {step['gate']}")
            print(f"  critic verdict      : {'ACCEPT' if step['accepted'] else 'REJECT'}")
            if step["critique"]:
                print(f"  critique            : {step['critique']}")


def main():
    ap = argparse.ArgumentParser(description="PragmaSense-AI demo")
    ap.add_argument("--text", default="ไปทานข้าวด้วยกันไหมครับ", help="Thai utterance")
    ap.add_argument("--context", default="", help="optional situation, e.g. 'employee to manager'")
    ap.add_argument("--gender", default="", choices=["", "male", "female"], help="speaker gender if not marked")
    ap.add_argument("--backend", default=None, help="gemini | openai | mock (default: $LLM_BACKEND or gemini)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--k_max", type=int, default=2)
    ap.add_argument("--mock", action="store_true", help="offline mock LLM for testing the code only")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    llm = LLMClient(backend="mock" if args.mock else args.backend, model=args.model)
    ps = PragmaSense(llm, k_max=args.k_max)

    print(LINE)
    print(" PRAGMASENSE-AI  |  Thai -> Myanmar pragmatic translation")
    print(f" model: {llm.backend}/{llm.model}" + ("   ** MOCK MODE - not a real translation **" if args.mock else ""))
    print(LINE)
    print(f"Input: {args.text}" + (f"   (situation: {args.context})" if args.context else ""))

    res = ps.translate(args.text, args.context, args.gender)
    show(res)
    print("\n" + "-" * 64)
    print(f"FINAL OUTPUT : {res.translation}")
    print(f"status       : {res.status} | revisions: {res.iterations} | LLM calls: {res.llm_calls} | "
          f"LLM time: {res.latency:.2f}s")
    print(LINE)


if __name__ == "__main__":
    main()
