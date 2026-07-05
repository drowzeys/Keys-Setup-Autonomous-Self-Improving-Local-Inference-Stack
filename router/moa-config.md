# Hermes MoA configuration — self-improving local stack

**Audience: the Hermes agent.** This is your configuration brief for the 4× DGX Spark
local inference stack. It tells you which agents exist, what each is good at, how to route,
and how the self-improvement / LoRA loop turns your own routing log into training data.
Goal: **keep ~90% of work on local silicon; escalate only the hard ~10% to cloud; capture
every escalation as a training label so the local share grows.**

## 1. Live agent roster (endpoints as of 2026-07-05)

| Agent | Endpoint | API | Competence | Relative cost | MoA role |
|---|---|---|---|---|---|
| **DSV4F · DSpark** | `http://10.100.10.1:8000/v1` | OpenAI | reasoning, routing, aggregation, tool-planning | med (local) | **Aggregator + default** |
| **Qwen3.6-27B-NVFP4** | `http://10.100.10.4:8000/v1` (`qwen36-nvfp4-a4q`) | OpenAI | light: classify / extract / summarize / rewrite / short chat; 256K ctx | **lowest** (local) | reference (light tier) |
| **Nemotron-3-Omni** | `http://10.100.10.3:8001/v1` | OpenAI | multimodal ingest (audio/vision/text), perception grounding | low (local) | reference (perception) |
| **True Two-Tower** | `http://10.100.10.3:8010/generate` | custom (not OpenAI) | gen-heavy / batch / repetitive diffusion generation | low (local) | **delegate target** (not a chat slot) |
| **Gemma-4-12B** | training node (rotates) | trainer | LoRA fine-tuning on omni-derived + cloud-gold pairs | n/a | **student / trainer** |
| **Cloud frontier** | codex `gpt-5.5` / opus-4.8 / grok | OpenAI | hard reasoning, audit, open-web research | **highest** (metered) | **escalation only** |

> Two-Tower speaks `/generate` (not `/v1/chat`), so it is **not** a standard MoA reference
> slot — route to it by direct delegation for gen-heavy/repetitive jobs (POST `{prompt,
> max_new_tokens}`). Everything else is OpenAI-compatible and slottable.

## 2. Desired MoA preset (replace the current all-cloud `default`)

Current `default` preset is cloud-only (codex + deepseek-v4-pro, aggregator opus) and **off**.
Reconfigure it local-first:

```
Reference models (the agents that draft):
  1. custom:qwen36-nvfp4-a4q         @ http://10.100.10.4:8000/v1   # light tier
  2. custom:nemotron-omni   @ http://10.100.10.3:8001/v1   # perception
  (optional 3.) custom:deepseek-v4-flash-dspark @ http://10.100.10.1:8000/v1  # heavy draft
Aggregator:
  custom:deepseek-v4-flash-dspark    @ http://10.100.10.1:8000/v1   # DSV4F merges/votes
Fallback / escalation (only when local confidence is low):
  openai-codex:gpt-5.5  /  openrouter:anthropic/claude-opus-4.8  /  grok
```

Apply with `hermes moa configure` (interactive) — pick the three local reference models and
set DSV4F as aggregator. Keep cloud in `hermes fallback`, not in the default reference set,
so it fires only on escalation.

## 3. Routing policy (cheapest competent agent)

1. **Perception first.** Any non-text input → Nemotron-Omni to produce structured text.
2. **Classify + route** to the cheapest agent that can satisfy the task:
   - trivial / light (classify, extract, summarize, rewrite, short chat) → **Qwen3.6**
   - gen-heavy / batch / repetitive → **delegate to Two-Tower `/generate`**
   - perception recall / grounding → **Nemotron-Omni**
   - multi-draft or ambiguous → **MoA**: fan out to reference agents, **DSV4F aggregates/votes**
   - genuinely hard / needs audit / needs open web → **escalate to cloud** (budget-gated)
3. **Target ≥90% local.** Escalate only the most novel/hard ~10% — those are also the most
   informative samples to learn from.

## 4. Self-improvement / LoRA loop (why you log everything)

You (DSV4F/Hermes) see every task, route, draft, and verdict — that log **is** the training set.

```
Hermes routing log ──▶ (task, chosen agent, candidate drafts, final verdict, cloud-gold?)
        │
        ▼  transcript miner selects high-value pairs:
        • cloud-corrected answers (local was wrong → gold exists)   ← highest value
        • high-agreement local wins (cheap self-distillation)
        • omni-derived pairs (Nemotron transcript ↔ desired output)
        │
        ▼
   Gemma-4-12B  ──LoRA fine-tune (omni-input aware)──▶ adapter Δ
        │
        ▼  hot-swap adapter into the serving specialist (Qwen / Omni / DSV4F draft head)
        ▼
   next same-class task → answered LOCALLY, no cloud call
```

**Consequence:** every cloud escalation buys a permanent local capability; the cloud share
trends down over time. Log routing decisions + verdicts to the training-signal store so the
miner has material. Prefer local aggregation (DSV4F voting across reference agents) over a
cloud call whenever the local drafts agree — that alone reclaims most would-be escalations.

## 5. Quick checks

- `hermes moa list` — see current slots
- `hermes moa configure` — set the local-first preset above
- `hermes fallback` — keep cloud here (escalation), not in the default reference set
- Health: `curl http://10.100.10.4:8000/v1/models`, `curl http://10.100.10.3:8001/health`,
  `curl http://10.100.10.3:8010/health`, `curl http://10.100.10.1:8000/v1/models`
