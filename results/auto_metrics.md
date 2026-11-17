# Automatic metrics

KB drift = % outputs rejected by the knowledge-base critic (diagnostic; PragmaSense optimises this).

| System | N | KB drift (%) | mean Δ | chrF++ | LLM calls / item | LLM time / item (s) |
|---|---|---|---|---|---|---|
| Google Translate | 32 | 43.8 | 0.66 | 36.2 | 0.00 | 0.00 |
| Zero-shot LLM | 32 | 28.1 | 0.41 | 35.2 | 1.00 | 2.19 |
| Chain-of-Thought | 32 | 9.4 | 0.20 | 39.5 | 1.00 | 3.50 |
| PragmaSense-AI | 32 | 3.1 | 0.16 | 37.0 | 1.28 | 2.84 |
| Human reference | 32 | 50.0 | 0.78 | 100.0 | 0.00 | 0.00 |

## KB drift (%) by archetype

| System | deference | softening | urgency | familiarity |
|---|---|---|---|---|
| Google Translate | 38 | 25 | 25 | 88 |
| Zero-shot LLM | 12 | 50 | 38 | 12 |
| Chain-of-Thought | 0 | 0 | 12 | 25 |
| PragmaSense-AI | 0 | 0 | 12 | 0 |
| Human reference | 50 | 0 | 88 | 62 |

## PragmaSense metacognitive loop

- drafts accepted without revision: 23/32
- drafts rejected by critic: 9/32
  - fixed by revision: 8 (88.9% of rejected)
  - stopped by gate (revision gave no gain): 1
  - still rejected after k_max revisions: 0
- mean revision rounds (rejected drafts only): 1.00
