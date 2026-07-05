# Routing-log hook — wiring the router path to the LoRA loop

Feeds `~/.hermes/routing_log.jsonl` (the miner's highest-value source) by capturing
each Hermes LLM call. `routing_log_hook.py` is installed at
`~/.hermes/hooks/routing_log.py` and declared in `~/.hermes/config.yaml`:

```yaml
hooks:
  post_llm_call:
    - matcher: ".*"
      command: /home/keyspark/.hermes/hooks/routing_log.py
      timeout: 10
```

**Status:** wired + validated (`hermes hooks list` shows it; `hermes hooks test
post_llm_call` fires it, exit 0). It is **non-blocking** — always exits 0 and writes
nothing to stdout, so it can never stall a Hermes turn; it skips silently when a turn
carries no useful (task, response) pair.

**Activation — needs one-time consent (by design):** `hooks_auto_accept` is left `false`
(safe). The first time Hermes makes an LLM call it will prompt once to allowlist
`routing_log.py`; approve it and logging runs automatically thereafter. We deliberately do
NOT flip `hooks_auto_accept: true`, which would let *any* hook run arbitrary commands
unreviewed.

**Payload caveat:** the synthetic `post_llm_call` payload exposes only
`extra:{model,platform}`; whether the live payload includes the task/response text is
unconfirmed on this build. The hook is defensive (logs whatever is present). If the live
payload turns out thin, the reliable fallback already exists: `train/mine_signal.py` also
mines Hermes' persisted transcripts (kanban outcomes + request dumps) for full pairs.

**Best signal:** for guaranteed-rich pairs, have the router append one line per decision:
`{"task","chosen_agent","drafts":[...],"verdict","cloud_gold":<text|null>}` — `cloud_gold`
(escalated/corrected answers) is weighted 2× in training.
