# Routing-log hook pipeline — wiring the router path to the LoRA loop

Feeds `~/.hermes/routing_log.jsonl` (the miner's highest-value source) by capturing
each Hermes LLM call with intelligent routing decisions.

## Architecture: two hooks, one pipeline

```
User prompt
    |
    v
[pre_llm_call] routing_router.py     <-- classifies task, picks cheapest competent agent
    |                                      writes routing decision to _last_routing.json
    v
Hermes LLM call (to chosen agent)
    |
    v
[post_llm_call] routing_log.py       <-- reads routing decision, captures task+verdict
    |                                      appends to routing_log.jsonl
    v
~/.hermes/routing_log.jsonl          <-- consumed by mine_signal.py -> LoRA training
```

## Files

| File | Role | Installed at |
|------|------|-------------|
| `routing_router.py` | pre_llm_call: classifies task, routes to cheapest competent agent | `~/.hermes/hooks/routing_router.py` |
| `routing_log.py` | post_llm_call: captures task+verdict, logs to JSONL | `~/.hermes/hooks/routing_log.py` |

## Routing policy (cheapest competent agent)

The classifier in `routing_router.py` implements the policy from `moa-config.md`:

| Category | Routed to | When |
|----------|-----------|------|
| **light** | Qwen3.6-27B (cheapest) | classify, extract, summarize, rewrite, short chat, <100 chars |
| **multimodal** | Nemotron-Omni | image/audio/video input |
| **gen_heavy** | DSV4F -> Two-Tower delegate | generate, produce, create, write, batch, bulk, render |
| **moa** | DSV4F (MoA fan-out) | complex/ambiguous tasks needing multi-draft |
| **escalation** | DSV4F + cloud fallback | audit, security, research, hard reasoning |

## Config declaration

In `~/.hermes/config.yaml`:

```yaml
hooks:
  pre_llm_call:
    - matcher: ".*"
      command: /home/keyspark/.hermes/hooks/routing_router.py
      timeout: 5
  post_llm_call:
    - matcher: ".*"
      command: /home/keyspark/.hermes/hooks/routing_log.py
      timeout: 10
```

## Activation

Hooks require one-time consent (by design — `hooks_auto_accept` is `false`).
Run `hermes chat --accept-hooks` once, or approve at the TTY prompt when prompted.

Verify with:
```bash
hermes hooks list          # should show both hooks as "allowed"
hermes hooks test pre_llm_call    # exit=0
hermes hooks test post_llm_call   # exit=0
```

## Routing log schema

Each line in `~/.hermes/routing_log.jsonl`:

```json
{
  "ts": "2026-07-05T14:11:44Z",
  "session_id": "abc123",
  "task": "user's input text",
  "chosen_agent": "qwen36-nvfp4-a4q",
  "route_type": "light",
  "drafts": [],
  "verdict": "assistant's response text",
  "cloud_gold": null
}
```

When a task is escalated to cloud, `cloud_gold` is set to `"cloud"` — this is
weighted 2x in training by `mine_signal.py`.

## LoRA training cycle

Once >=50 pairs accumulate in `routing_log.jsonl`:

```bash
~/keys-setup-stack/train/run-lora-cycle.sh
```

This runs `mine_signal.py` -> `gemma_lora_train.py` -> eval-gate -> hot-swap adapter.
