#!/usr/bin/env python3
"""
Hermes pre_llm_call hook: intelligent MoA routing.

Implements the "cheapest competent agent" policy from moa-config.md:
  - trivial/light -> Qwen3.6-27B (cheapest)
  - multimodal input -> Nemotron-Omni (perception)
  - gen-heavy/batch -> Two-Tower /generate (delegate)
  - multi-draft/ambiguous -> MoA fan-out (DSV4F aggregates)
  - genuinely hard -> escalate to cloud fallback

Also logs every routing decision to ~/.hermes/routing_log.jsonl
for the LoRA self-improvement loop.

Runs as a pre_llm_call hook. Reads the payload on stdin, decides the
best agent, and writes the decision to a shared state file that the
post_llm_call hook can pick up.
"""
import sys, json, os, re, datetime, datetime as dt

STATE_FILE = os.path.expanduser("~/.hermes/hooks/_last_routing.json")
LOG_FILE = os.path.expanduser("~/.hermes/routing_log.jsonl")

# Agent definitions from moa-config.md
AGENTS = {
    "qwen36-nvfp4-a4q": {
        "endpoint": "http://10.100.10.4:8000/v1",
        "cost": 1,       # cheapest
        "strengths": ["classify", "extract", "summarize", "rewrite", "short_chat", "light"],
        "max_ctx": 256000,
    },
    "nemotron-omni": {
        "endpoint": "http://10.100.10.3:8001/v1",
        "cost": 2,       # medium
        "strengths": ["multimodal", "perception", "vision", "audio", "grounding"],
        "max_ctx": 128000,
    },
    "deepseek-v4-flash-dspark": {
        "endpoint": "http://10.100.10.1:8000/v1",
        "cost": 3,       # higher
        "strengths": ["reasoning", "routing", "aggregation", "tool_planning", "complex"],
        "max_ctx": 128000,
    },
}

CLOUD_AGENTS = {
    "codex:gpt-5.5": {"cost": 10, "strength": "hard_reasoning"},
    "anthropic/claude-opus-4.8": {"cost": 12, "strength": "audit"},
    "grok": {"cost": 8, "strength": "research"},
}

# Lightweight task classifier keywords
LIGHT_PATTERNS = re.compile(
    r"^(classif|extract|summar|rewrit|short|transl|format|pars|convert|list|"
    r"what is|who is|define|explain briefly|yes|no|true|false|hello|hi)\b",
    re.IGNORECASE,
)

# If the entire task is under 100 chars and doesn't match other patterns, it's light
LIGHT_LENGTH_THRESHOLD = 100

MULTIMODAL_PATTERNS = re.compile(
    r"\b(image|picture|photo|screenshot|audio|video|speech|voice|see|look|describe.*(image|photo|screen))\b",
    re.IGNORECASE,
)

GEN_HEAVY_PATTERNS = re.compile(
    r"\b(generat\w*|produc\w*|creat\w*|write\w*|batch|bulk|many|repetitive|diffus\w*|render\w*)\b",
    re.IGNORECASE,
)

HARD_PATTERNS = re.compile(
    r"\b(audit|security|vulnerab|exploit|research|deep.*(analys|review)|"
    r"complex|difficult|hard|novel|unprecedented)\b",
    re.IGNORECASE,
)


def classify_task(task_text):
    """Classify a task into a routing category."""
    if not task_text:
        return "light"  # default safe

    first_line = task_text.split("\n")[0].strip()

    # Check for gen-heavy FIRST (unconditional -- these patterns override length-based light)
    if GEN_HEAVY_PATTERNS.search(task_text):
        return "gen_heavy"

    # Check for hard/complex
    if HARD_PATTERNS.search(task_text):
        return "hard"

    # Check for multimodal
    if MULTIMODAL_PATTERNS.search(task_text):
        return "multimodal"

    # Check for light/trivial (short tasks or matching keywords)
    if LIGHT_PATTERNS.match(first_line) or len(task_text) < LIGHT_LENGTH_THRESHOLD:
        return "light"

    # Default: multi-draft / ambiguous -> MoA
    return "moa"


def route_task(task_text, has_multimodal_input=False):
    """
    Determine the best agent for a task.
    Returns (chosen_agent, route_type, cloud_gold).
    """
    if has_multimodal_input:
        return "nemotron-omni", "perception", None

    category = classify_task(task_text)

    if category == "light":
        return "qwen36-nvfp4-a4q", "light", None

    if category == "multimodal":
        return "nemotron-omni", "perception", None

    if category == "gen_heavy":
        # Two-Tower is a delegate target, not a chat slot
        # For now, route to DSV4F which can delegate to Two-Tower
        return "deepseek-v4-flash-dspark", "gen_heavy", None

    if category == "hard":
        # Escalate to cloud -- this creates a cloud_gold training signal
        return "deepseek-v4-flash-dspark", "escalation", "cloud"

    # Default: MoA fan-out
    return "deepseek-v4-flash-dspark", "moa", None


def log_decision(task, chosen_agent, route_type, cloud_gold, session_id=None):
    """Append a routing decision to the JSONL log."""
    row = {
        "ts": datetime.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
        "task": task[:2000] if task else "",
        "chosen_agent": chosen_agent,
        "route_type": route_type,
        "drafts": [],
        "verdict": None,
        "cloud_gold": cloud_gold,
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    extra = payload.get("extra", {})
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = {}

    # Extract the task -- real Hermes payload uses extra.user_message
    task = extra.get("user_message") or extra.get("prompt") or extra.get("task") or ""
    has_multimodal = False

    # Also check messages array format (for synthetic/test payloads)
    if not task:
        messages = extra.get("messages") or payload.get("messages") or []
        if isinstance(messages, list):
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content", "")
                    if isinstance(content, list):
                        has_multimodal = any(
                            isinstance(p, dict) and p.get("type") in ("image_url", "image", "audio")
                            for p in content
                        )
                        task = json.dumps(content)[:2000]
                    elif isinstance(content, str):
                        task = content[:2000]
                    break

    if not task:
        task = payload.get("prompt", "")

    # Route
    chosen_agent, route_type, cloud_gold = route_task(task, has_multimodal)

    session_id = payload.get("session_id")

    # Save routing decision for post_llm_call hook
    decision = {
        "chosen_agent": chosen_agent,
        "route_type": route_type,
        "cloud_gold": cloud_gold,
        "ts": datetime.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(decision, f)
    except Exception:
        pass

    # Log to routing_log.jsonl
    log_decision(task, chosen_agent, route_type, cloud_gold, session_id)

    # Print nothing to stdout (hooks must be silent)


if __name__ == "__main__":
    main()
