#!/usr/bin/env python3
"""Hermes post_llm_call hook -> routing_log.jsonl for the LoRA self-improvement loop.

Reads the hook payload on stdin and appends ONE line per LLM call in the schema
mine_signal.py expects:
  {"task","chosen_agent","drafts","verdict","cloud_gold","ts","session_id"}

Defensive: the payload envelope is {hook_event_name,tool_name,tool_input,session_id,
cwd,extra}. The call specifics live in `extra` (dict). We pull the model as
chosen_agent, the last user message as task, and the assistant text as verdict,
searching a few likely key names so it survives payload variations. Never blocks:
always exits 0 and prints nothing to stdout.
"""
import sys, json, os, datetime

LOG = os.path.expanduser("~/.hermes/routing_log.jsonl")
DBG = os.path.expanduser("~/.hermes/hooks/last_payload.json")

def dig(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, "", []):
            return d[k]
    return None

def last_user(messages):
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                return c if isinstance(c, str) else json.dumps(c)[:8000]
    return None

def text_of(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return dig(x, "content", "text") or (
            (x.get("choices", [{}])[0].get("message", {}) or {}).get("content"))
    return None

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

    model = dig(extra, "model") or dig(p, "model") or "unknown"
    messages = dig(extra, "messages", "input_messages", "request_messages") \
        or dig(p, "messages")
    task = last_user(messages) or dig(extra, "prompt", "task", "user_prompt")
    verdict = text_of(dig(extra, "response", "completion", "output", "assistant")) \
        or dig(extra, "response_text", "content")

    if not task and not verdict:
        return  # nothing useful this call (e.g. tool-only turn) — skip silently

    row = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "session_id": p.get("session_id"),
        "task": task,
        "chosen_agent": model,
        "drafts": [],          # populated only when MoA fan-out is used
        "verdict": verdict,
        "cloud_gold": None,    # set by the escalation path when a cloud model is used
    }
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
