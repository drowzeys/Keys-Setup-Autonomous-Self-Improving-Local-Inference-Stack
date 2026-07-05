#!/usr/bin/env python3
"""Mine Hermes' own history into LoRA training pairs for the self-improvement loop.

Sources (whatever is present):
  1. ~/.hermes/kanban.db  — completed tasks (title+body -> resolution) = task->outcome pairs
  2. ~/.hermes/sessions/*request_dump*.json — captured request/response
  3. ~/.hermes/routing_log.jsonl — the forward schema the router SHOULD emit per decision:
       {"task","chosen_agent","drafts":[...],"verdict","cloud_gold":<text|null>,"ts"}

Emits train_pairs.jsonl: {"messages":[{role,content}...], "source", "weight", "kind"}
  kind = routing | aggregation | specialist | task_outcome
Highest weight to cloud-corrected pairs (gold the local stack got wrong) and to
routing decisions (these improve the MoA LOGIC itself, not just leaf answers).
"""
import argparse, glob, json, os, sqlite3, sys

HOME = os.path.expanduser("~")
OUT = []

def add(messages, source, kind, weight):
    if not messages or not any(m.get("content") for m in messages):
        return
    OUT.append({"messages": messages, "source": source, "kind": kind, "weight": weight})

def mine_kanban(path):
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for tid, title, body, status in c.execute(
                "select id,title,body,status from tasks"):
            if (status or "").lower() in ("done", "completed", "closed") and title:
                # task -> resolution; the resolution teaches task decomposition
                res = ""
                try:
                    rows = c.execute(
                        "select body from task_comments where task_id=? order by created_at",
                        (tid,)).fetchall()
                    res = "\n".join(r[0] for r in rows if r and r[0])[:4000]
                except Exception:
                    res = ""
                out = res or (body or "")
                add([{"role": "user", "content": (title + "\n\n" + (body or "")).strip()},
                     {"role": "assistant", "content": out.strip()}],
                    "kanban", "task_outcome", 1.0)
                n += 1
    except Exception as e:
        print(f"[mine] kanban skipped: {e}", file=sys.stderr)
    return n

def mine_request_dumps(globpat):
    n = 0
    for f in glob.glob(globpat):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # best-effort: look for a messages list + a final assistant reply
        msgs = d.get("messages") or d.get("request", {}).get("messages")
        if isinstance(msgs, list) and msgs:
            reply = d.get("response") or d.get("completion")
            if isinstance(reply, dict):
                reply = (reply.get("choices", [{}])[0].get("message", {}) or {}).get("content")
            if reply:
                add(msgs + [{"role": "assistant", "content": str(reply)}],
                    "request_dump", "specialist", 0.8)
                n += 1
    return n

def mine_routing_log(path):
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        task = r.get("task")
        gold = r.get("cloud_gold")
        if not task:
            continue
        if gold:  # local was escalated/corrected -> highest-value gold pair
            add([{"role": "user", "content": task},
                 {"role": "assistant", "content": str(gold)}],
                "routing_log", "specialist", 2.0)
            n += 1
        # routing-decision pair: teach WHICH agent handles this task (improves MoA logic)
        agent = r.get("chosen_agent")
        if agent:
            add([{"role": "user",
                  "content": f"Route this task to the cheapest competent agent.\nTASK: {task}"},
                 {"role": "assistant", "content": f"route: {agent}"}],
                "routing_log", "routing", 1.5)
            n += 1
        # aggregation preference: which draft won (improves the aggregator)
        drafts, verdict = r.get("drafts"), r.get("verdict")
        if isinstance(drafts, list) and len(drafts) > 1 and verdict:
            add([{"role": "user",
                  "content": "Pick the best answer for:\n" + task + "\n\nCANDIDATES:\n" +
                             "\n---\n".join(str(x) for x in drafts)},
                 {"role": "assistant", "content": str(verdict)}],
                "routing_log", "aggregation", 1.2)
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HOME, "keys-setup-stack/train/train_pairs.jsonl"))
    a = ap.parse_args()
    k = mine_kanban(os.path.join(HOME, ".hermes/kanban.db"))
    r = mine_request_dumps(os.path.join(HOME, ".hermes/sessions/*request_dump*.json"))
    g = mine_routing_log(os.path.join(HOME, ".hermes/routing_log.jsonl"))
    with open(a.out, "w") as f:
        for row in OUT:
            f.write(json.dumps(row) + "\n")
    by = {}
    for row in OUT:
        by[row["kind"]] = by.get(row["kind"], 0) + 1
    print(f"[mine] kanban={k} request_dumps={r} routing_log={g} "
          f"-> {len(OUT)} pairs {by} -> {a.out}")

if __name__ == "__main__":
    main()
