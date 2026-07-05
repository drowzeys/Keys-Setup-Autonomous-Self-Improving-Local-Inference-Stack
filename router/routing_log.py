#!/usr/bin/env python3
"""Hermes post_llm_call hook -> routing_log.jsonl for the LoRA self-improvement loop.

Reads the hook payload on stdin and appends ONE line per LLM call in the schema
mine_signal.py expects:
  {"task","chosen_agent","drafts","verdict","cloud_gold","ts","session_id"}

Real Hermes payload format (from extra):
  user_message -> task
  assistant_response -> verdict
  model -> chosen_agent

Also reads the routing decision left by the pre_llm_call hook (routing_router.py).
Never blocks: always exits 0 and prints nothing to stdout.
"""
import sys, json, os, datetime

LOG = os.path.expanduser("~/.hermes/routing_log.jsonl")
DBG = os.path.expanduser("~/.hermes/hooks/last_payload.json")
ROUTING_STATE = os.path.expanduser("~/.hermes/hooks/_last_routing.json")


def main():
    raw = sys.stdin.read()
    try:
        p = json.loads(raw) if raw.strip() else {}
    except Exception:
        p = {}
    try:
        open(DBG, "w").write(raw)  # keep last payload for inspection
    except Exception:
        pass

    extra = p.get("extra")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra.replace("'", '"'))
        except Exception:
            extra = {}
    extra = extra or {}

    # Read routing decision from pre_llm_call hook
    routing = {}
    try:
        routing = json.load(open(ROUTING_STATE))
    except Exception:
        pass

    # Real Hermes payload: extra.user_message, extra.assistant_response, extra.model
    task = extra.get("user_message") or extra.get("prompt") or extra.get("task") or ""
    verdict = extra.get("assistant_response") or extra.get("response") or extra.get("completion") or ""
    model = routing.get("chosen_agent") or extra.get("model") or "unknown"
    route_type = routing.get("route_type", "unknown")
    cloud_gold = routing.get("cloud_gold")

    # Fallback: check messages array (for synthetic/test payloads)
    if not task or not verdict:
        messages = extra.get("messages") or p.get("messages") or []
        if isinstance(messages, list):
            for m in reversed(messages):
                if isinstance(m, dict):
                    if m.get("role") == "user" and not task:
                        c = m.get("content", "")
                        task = c if isinstance(c, str) else json.dumps(c)[:8000]
                    if m.get("role") == "assistant" and not verdict:
                        c = m.get("content", "")
                        verdict = c if isinstance(c, str) else json.dumps(c)[:8000]

    if not task and not verdict:
        return  # nothing useful this call (e.g. tool-only turn) -- skip silently

    row = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "session_id": p.get("session_id"),
        "task": task,
        "chosen_agent": model,
        "route_type": route_type,
        "drafts": [],          # populated only when MoA fan-out is used
        "verdict": verdict,
        "cloud_gold": cloud_gold,    # set by the routing hook when escalated
    }
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
