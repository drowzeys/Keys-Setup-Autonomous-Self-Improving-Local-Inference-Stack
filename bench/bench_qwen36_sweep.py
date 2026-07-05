#!/usr/bin/env python3
"""Qwen3.6-27B-NVFP4 A4Q bench on node .4 :8000 (model qwen36-nvfp4-a4q).
Reuses bench_exp1_sweep methodology: OpenAI streaming + include_usage,
measures TTFT + per-request decode tok/s + aggregate tok/s.

Modes:
  warm                         one request to hotten JIT/graphs
  conc                         concurrency sweep C in {1,2,4,8,16,20}, ~512-tok in / 128 out
  ctx  <tag> <K,csv>           single-stream TTFT at given prompt sizes (K tokens), 128 out
  coh                          one short factual prompt (coherence parity)
"""
import json, sys, time, threading, urllib.request

BASE = "http://r0:8000/v1/chat/completions"
MODEL = "qwen36-nvfp4-a4q"

def stream_request(messages, max_tokens, temperature=0.0, timeout=3600):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE, data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None
    text = []; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                c = delta.get("content") or delta.get("reasoning_content") or ""
                if c:
                    if t_first is None:
                        t_first = time.time()
                    t_last = time.time()
                    text.append(c)
    ttft = (t_first - t0) if t_first else None
    ct = usage.get("completion_tokens") if usage else None
    pt = usage.get("prompt_tokens") if usage else None
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    tps = (ct - 1) / decode_s if (ct and decode_s) else None
    return {"ttft": ttft, "completion_tokens": ct, "prompt_tokens": pt,
            "decode_tps": tps, "wall": time.time() - t0, "text": "".join(text)}

# ~512-token prompt (measured prompt_tokens reported in results)
FILLER = ("The grain harvest in the valley proceeded on schedule that season, and the "
    "millers recorded each delivery in heavy canvas ledgers. Wagons arrived from the "
    "eastern farms before noon, their axles creaking under sacks of barley and rye. "
    "The weighing house stood beside the river, where the current turned the great "
    "stone wheels day and night. Children gathered near the loading dock to watch the "
    "teamsters stack the sacks in tidy pyramids. ")
TOK_PER_REPEAT = 88.06  # calibrated earlier on this tokenizer

def build_prompt(target_tok):
    n = max(1, round((target_tok - 40) / TOK_PER_REPEAT))
    return ("Read the following report, then write a concise 120-word summary of it.\n\n"
            + FILLER * n + "\n\nNow write the 120-word summary.")

CONC_PROMPT = build_prompt(512)

def conc_point(C, stagger=0.03):
    results = [None] * C
    errors = [None] * C
    def worker(i):
        try:
            results[i] = stream_request(
                [{"role": "user", "content": CONC_PROMPT + f" (variant {i})"}],
                128, temperature=0.7)
        except Exception as e:
            errors[i] = str(e)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    t0 = time.time()
    for t in threads:
        t.start()
        if stagger:
            time.sleep(stagger)  # staggered start
    for t in threads: t.join()
    wall = time.time() - t0
    good = [r for r in results if r]
    total_ct = sum(r["completion_tokens"] or 0 for r in good)
    ttfts = [r["ttft"] for r in good if r["ttft"]]
    first = min(ttfts) if ttfts else 0
    agg = total_ct / (wall - first) if wall > first else None
    dec = [r["decode_tps"] for r in good if r["decode_tps"]]
    mean_pt = sum(r["prompt_tokens"] or 0 for r in good) / max(1, len(good))
    return {"C": C, "agg_tps": round(agg, 2) if agg else None,
            "mean_req_decode_tps": round(sum(dec) / max(1, len(dec)), 2),
            "mean_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            "mean_prompt_tok": round(mean_pt, 1),
            "total_completion_tok": total_ct, "wall_s": round(wall, 2),
            "n_errors": sum(1 for e in errors if e),
            "errors": [e for e in errors if e][:3]}

# context sweep for TTFT A/B
CTX_FILLER = FILLER
def build_ctx_prompt(target_k):
    target = target_k * 1000
    n = max(4, int((target - 60) / TOK_PER_REPEAT))
    at = int(n * 0.75)
    parts = []
    for i in range(n):
        if i == at:
            parts.append("The passkey is 7429. ")
        parts.append(CTX_FILLER)
    parts.append("\n\nWhat is the passkey? Reply with just the number.")
    return "".join(parts)

def ctx_point(target_k):
    prompt = build_ctx_prompt(target_k)
    r = stream_request([{"role": "user", "content": prompt}], 128, temperature=0.0)
    return {"target_k": target_k, "prompt_tokens": r["prompt_tokens"],
            "ttft_s": round(r["ttft"], 3) if r["ttft"] else None,
            "decode_tps": round(r["decode_tps"], 2) if r["decode_tps"] else None,
            "completion_tokens": r["completion_tokens"],
            "passkey_ok": "7429" in r["text"],
            "tail": r["text"][-100:].replace("\n", " ")}

COH_Q = "In one sentence, what is the capital of Australia and on which continent is it?"

def main():
    mode = sys.argv[1]
    if mode == "warm":
        r = stream_request([{"role": "user", "content": "Say hello in one short sentence."}], 32)
        print("warm ttft", round(r["ttft"], 3), "text:", r["text"][:80].replace("\n", " "))
    elif mode == "conc":
        out = {"mode": "conc", "model": MODEL, "prompt_target_tok": 512,
               "max_tokens": 128, "points": []}
        for C in [1, 2, 4, 8, 16, 20]:
            print(f"=== conc C={C} ===", flush=True)
            p = conc_point(C)
            out["points"].append(p)
            print(json.dumps(p), flush=True)
            time.sleep(3)
        json.dump(out, open("/home/keyspark/a4q-lab/qwen36_conc_sweep.json", "w"), indent=2)
        print("saved qwen36_conc_sweep.json")
    elif mode == "ctx":
        tag = sys.argv[2]
        pts = [int(x) for x in sys.argv[3].split(",")]
        out = {"mode": "ctx", "tag": tag, "model": MODEL, "points": []}
        for k in pts:
            print(f"=== ctx {k}K (tag={tag}) ===", flush=True)
            p = ctx_point(k)
            out["points"].append(p)
            print(json.dumps(p), flush=True)
            time.sleep(2)
        json.dump(out, open(f"/home/keyspark/a4q-lab/qwen36_ctx_{tag}.json", "w"), indent=2)
        print(f"saved qwen36_ctx_{tag}.json")
    elif mode == "coh":
        r = stream_request([{"role": "user", "content": COH_Q}], 64, temperature=0.0)
        print(json.dumps({"q": COH_Q, "text": r["text"].strip(),
                          "ttft_s": round(r["ttft"], 3) if r["ttft"] else None}))

if __name__ == "__main__":
    main()
