# PragmaSense-AI

A dual-process (System 1 / System 2) cognitive pipeline for preserving politeness and tone
(sentence-final particles) in Thai → Myanmar spoken-dialogue translation.
CPE 663 Cognitive Computing term project, KMUTT.

```
Thai utterance
   │
   ▼
Tier 1  Perception ── extracts Thai particles (ครับ, นะ, สิ, วะ, …) ──► target register level I_t + functions
   │                     (symbolic knowledge base: data/knowledge_base.json)
   ▼
Tier 2  Generator (System 1) ── fast LLM draft translation H0
   │
   ▼
Tier 3  Critic (System 2) ── detects Myanmar markers (ပါ, ခင်ဗျာ, နော်, လေ, ကွာ, …) ──► I_g
   │       Δ = |I_t − I_g|; accept if Δ ≤ τ (0.5), no missing function, no gender mismatch
   │       otherwise: natural-language critique ──► revision (max k_max = 2, gated: a
   │       revision is kept only if it improves the critic score)
   ▼
Final translation + full reasoning trace
```

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/maythuthukhine3042-cyber/PragmaSense-AI.git
cd PragmaSense-AI
pip install -r requirements.txt
```

Get a free Gemini API key at https://aistudio.google.com ("Get API key"), then:

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
```

Optional: `export GEMINI_MODEL="<model name from AI Studio>"` (default `gemini-flash-latest`).
To use an OpenAI-compatible API instead: `export LLM_BACKEND=openai OPENAI_API_KEY=...`.

## Usage

```bash
# 1. Demo: one sentence with the full cognitive trace
python run_pipeline.py --text "ไปทานข้าวด้วยกันไหมครับ"
python run_pipeline.py --text "หิวว่ะ" --context "two close male friends"

# 2. Check the knowledge base on the human references
python check_kb.py

# 3. Run all systems on the benchmark (resumable; calls are cached)
python run_experiments.py --limit 3          # quick test first
python run_experiments.py --sleep 4          # full run: zero_shot, cot, pragmasense
# optional ablations: --systems zero_shot,cot,kb_no_critic,self_refine,pragmasense

# 4. Automatic metrics + blind sheet for human raters
python evaluate.py

# 5. After raters fill results/ratings_<name>.csv
python analyze_human.py
```

`--mock` runs the code offline with a fake LLM; it is only for testing the code and
produces no real translations.

## Systems compared

| Name | Description |
|---|---|
| `google_translate` | outputs pasted manually into `data/benchmark.csv` |
| `zero_shot` | single prompt, no pragmatic guidance (same prompt as the System-1 draft) |
| `cot` | chain-of-thought prompt that asks the model to reason about politeness first |
| `self_refine` | (optional ablation) LLM critiques and revises itself once, without the knowledge base |
| `kb_no_critic` | (optional ablation) knowledge-base hints in the prompt, single pass, no critic |
| `pragmasense` | full pipeline: draft → KB critic → gated revisions |

## Files

```
pragmasense/kb.py        knowledge base, Thai particle extraction, Myanmar marker detection
pragmasense/critic.py    Δ computation and critique generation
pragmasense/pipeline.py  PragmaSense pipeline and comparison systems
pragmasense/prompts.py   all prompts
pragmasense/llm.py       Gemini / OpenAI-compatible / mock client with cache
pragmasense/metrics.py   chrF++, weighted kappa, Spearman
data/knowledge_base.json particle table (levels 1–4 assigned by the author)
data/benchmark.csv       32 Thai–Myanmar pairs, 8 per archetype (benchmark_full48.csv = larger optional set)
```

## Limitations

The register levels in the knowledge base and the reference translations were assigned by
one native Myanmar speaker. Marker detection is rule-based (regex) and can miss or
over-detect markers. The KB-based drift rate is the same signal the critic optimises, so
blind human ratings are the main evaluation.
