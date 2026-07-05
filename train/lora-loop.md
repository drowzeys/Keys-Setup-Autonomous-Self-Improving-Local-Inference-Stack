# The LoRA self-improvement loop — Gemma-4-12B trains the whole MoA logic

**This is the engine that makes the stack _self-improving_.** Gemma-4-12B-it is the STUDENT.
It does not just learn better leaf answers — it learns the **MoA logic itself**: which agent
should handle which task (routing) and which draft wins (aggregation). Training Gemma and
folding its adapters back in makes the *whole stack* better over time, so the local share of
work keeps growing and the cloud bill keeps shrinking.

## Why Gemma-4-12B is the trainer
- Small enough to LoRA-fine-tune fast on one GB10 (23 GB bf16; r=16 adapter is cheap).
- **Omni-input aware** (it has a processor/vision head) → multimodal lessons from Nemotron-Omni flow in.
- Strong enough to absorb frontier (cloud) corrections and distill them into the local tier.

## The loop (Hermes can run it: `train/run-lora-cycle.sh`)
```
Hermes/DSV4F routing log  ──▶  mine_signal.py  ──▶  train_pairs.jsonl
   (task, chosen_agent, drafts, verdict, cloud_gold?)         │
                                                              ▼
                          gemma_lora_train.py (peft/trl LoRA on Gemma-4-12B)
                                                              │
                                                              ▼
                          adapter Δ  ──eval-gate──▶  hot-swap into serving specialist
                                                              │
                                                              ▼
                          next same-class task answered LOCALLY (no cloud)
```

## What gets mined — four pair kinds (weights up-rank MoA-logic + gold)
| kind | built from | teaches | weight |
|---|---|---|--:|
| **specialist** | cloud-gold answers, local wins, request dumps | better leaf answers | 0.8–2.0 (gold highest) |
| **routing** | `chosen_agent` per task | **which agent handles which task → improves MoA routing logic** | 1.5 |
| **aggregation** | `drafts` + `verdict` | **which draft the aggregator should pick → improves voting logic** | 1.2 |
| **task_outcome** | completed kanban tasks | task decomposition | 1.0 |

The **routing** and **aggregation** kinds are the point of "improving the whole MoA logic":
the router's own past decisions and the aggregator's own past verdicts become supervised
targets, so DSV4F's routing/aggregation heads get sharper — not just the specialists.

## Feeding the loop: the routing-log schema the stack should emit
For the highest-value signal, have the router append one line per decision to
`~/.hermes/routing_log.jsonl`:
```json
{"task": "...", "chosen_agent": "qwen36-nvfp4-a4q", "drafts": ["...","..."],
 "verdict": "the chosen/merged answer", "cloud_gold": null, "ts": "..."}
```
Set `cloud_gold` to the cloud answer whenever a task was escalated — that is the gold the
local stack failed on, and it is weighted 2× in training.

**Status: LIVE.** The two-hook pipeline (`routing_router.py` pre_llm_call +
`routing_log.py` post_llm_call) is installed, allowlisted, and firing. Every Hermes
LLM call is now logged to `~/.hermes/routing_log.jsonl` with task, verdict, chosen agent,
route type, and cloud_gold flag. The log is ready for `mine_signal.py` consumption.

## Cadence, eval-gate, rollback
- **Trigger:** Hermes runs a cycle nightly (cron) or after ≥`MIN_PAIRS` (default 50) new pairs.
- **Eval-gate:** never hot-swap blindly — run a held-out eval; promote the adapter only if it
  matches/beats the current one. Keep the previous adapter for instant rollback.
- **Hot-swap:** serve Gemma+adapter, or merge the adapter into the target specialist (Qwen light
  tier / Omni / DSV4F draft head) and repoint the router. No full redeploy.

## Files
- `mine_signal.py` — mine Hermes history → `train_pairs.jsonl` (runs on base python; works today).
- `gemma_lora_train.py` — peft/trl LoRA SFT on Gemma-4-12B (runs in a container with peft/trl).
- `run-lora-cycle.sh` — mine → train → gate/stage, one command for Hermes/cron.
