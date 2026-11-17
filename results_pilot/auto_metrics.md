# Automatic metrics

KB drift = % outputs rejected by the knowledge-base critic (diagnostic; PragmaSense optimises this).

| System | N | KB drift (%) | mean Δ | chrF++ | LLM calls / item | LLM time / item (s) |
|---|---|---|---|---|---|---|
| Google Translate | 32 | 65.6 | 1.05 | 39.1 | 0.00 | 0.00 |
| Zero-shot LLM | 32 | 18.8 | 0.25 | 51.8 | 1.00 | 2.07 |
| Chain-of-Thought | 32 | 28.1 | 0.28 | 51.0 | 1.00 | 3.06 |
| PragmaSense-AI | 32 | 3.1 | 0.16 | 52.7 | 1.19 | 2.41 |
| Human reference | 32 | 0.0 | 0.09 | 100.0 | 0.00 | 0.00 |

## KB drift (%) by archetype

| System | deference | softening | urgency | familiarity |
|---|---|---|---|---|
| Google Translate | 50 | 25 | 100 | 88 |
| Zero-shot LLM | 0 | 25 | 38 | 12 |
| Chain-of-Thought | 0 | 25 | 50 | 38 |
| PragmaSense-AI | 0 | 0 | 12 | 0 |
| Human reference | 0 | 0 | 0 | 0 |

## PragmaSense metacognitive loop

- drafts accepted without revision: 26/32
- drafts rejected by critic: 6/32
  - fixed by revision: 5 (83.3% of rejected)
  - stopped by gate (revision gave no gain): 1
  - still rejected after k_max revisions: 0
- mean revision rounds (rejected drafts only): 1.00
