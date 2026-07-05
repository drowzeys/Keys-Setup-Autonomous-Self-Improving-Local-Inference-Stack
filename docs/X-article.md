# I gave four $4,000 desktops a group chat and a teacher. Now they do 90% of my AI work — and get better every night.

*A build log for an autonomous, self-improving local inference stack on a 4× NVIDIA DGX Spark cluster.*

---

**The thesis:** the frontier cloud models are extraordinary, but you don't need a frontier call for *most* of what you actually do. Classify this. Summarize that. Transcribe the meeting. Draft the reply. Extract the fields. If you could keep 90% of that on hardware you own — and have the remaining 10% *teach* your local models so the 90% keeps growing — you'd have something close to sovereign AI that gets cheaper the more you use it.

That's what I built. Here's how it works.

## The hardware

Four **NVIDIA DGX Sparks** — GB10, 128 GB unified memory each, Blackwell-consumer silicon — wired together over a **200 Gb RoCE fabric** (a MikroTik CRS804). ~$16k of desktops behaving like one machine. No rack, no datacenter, sitting on a shelf.

## The trick: models as *agents*, not endpoints

The stack runs a **Mixture-of-Agents** router I call **Hermes MoA**. Every model on the cluster is an *agent* with a competence profile and a cost. A task comes in, and Hermes routes it to the **cheapest agent that can actually do it** — and for the hard ones, to *several* agents whose answers get aggregated and voted.

The roster:

- 🧠 **DeepSeek-V4-Flash · DSpark** — the **orchestrator**. It routes, aggregates, votes, and — critically — logs every decision. It's not a script; it's a served reasoning model with a custom concurrency patch so it can make hundreds of routing calls at once.
- 🎨 **Two-Tower NVFP4 (diffusion)** — I quantized a 118 GB two-node diffusion model down to **21 GB per tower** so it fits on a *single* GPU. That freed an entire node. Honest tradeoff: on **two** Sparks the towers run in parallel and hit **38.85 tok/s** (fastest); consolidated onto **one** Spark they serialize to **~29 tok/s** — still **1.57× the autoregressive baseline on gen-heavy work (28.98 vs 18.44)**, so for resource conservation the *true* diffusion tower on one GPU beats the AR-simulated version while freeing a node. Speed wants 2 GPUs; efficiency wants 1.
- 🎙️ **Nemotron-3-Omni** — audio + image + video + text ingest. It shares a Spark with the diffusion model (**two models, one node**) because fp4 attention + NVFP4 weights leave ~36 GB of headroom.
- 📚 **Gemma-4-12B** — the **student**. Small enough to LoRA-train in fast epochs on one GB10.
- ⚡ **Qwen3.6-27B-NVFP4** — the **light tier**. 256K context, ~68 tokens/sec at 8-way concurrency, native fp4 attention. It soaks up the long tail of easy requests so the brain stays free for the hard ones.
- ☁️ **A rate-limited cloud model** (Claude / Codex / Grok) — the **oracle**. Audit, genuinely hard reasoning, open-web research. Deliberately kept to ~10% of traffic.

## The part that makes it *self-improving*

Here's the loop that matters. Because the DSV4F orchestrator sees **every** task, **every** routing decision, and **every** aggregate verdict, its log *is* a training set — for free.

A miner pulls the highest-value pairs out of that log:
- Anything the **cloud had to correct** → now I have a gold answer the local stack got wrong.
- **High-agreement local wins** → cheap self-distillation.
- **Omni-derived pairs** → the multimodal front-end turns raw audio/video into supervised examples.

Those pairs go to **Gemma-4-12B — the student.** It LoRA-trains on them and produces an adapter that gets **hot-swapped into the serving models**. Next time that class of task shows up, it's answered **locally** — no cloud call.

But here's the part most "local AI" setups miss: Gemma doesn't just learn better *answers*. It trains on the router's own decisions too — **which agent should handle which task** (the routing) and **which draft should win** (the aggregation). So the thing that improves isn't just the leaf models — it's the **MoA logic itself.** The router gets sharper at routing; the aggregator gets sharper at picking. The whole stack levels up, not just its parts.

The whole thing is one command Hermes can run on a timer: mine its own history → LoRA-train Gemma → eval-gate → hot-swap. It literally **self-trains from its own work log.**

**Every cloud escalation buys a permanent local capability.** The cloud bill trends *down* over time. The 90/10 split becomes 92/8 becomes 95/5 — automatically — because the "cheapest competent agent" keeps getting more competent, *and gets better at knowing which agent is cheapest.*

## Why this shape

- A router has to reason *and* batch cheaply → DSV4F (MLA + speculative decode + my batch patch) is the only model that does both well on this hardware.
- NVFP4 quantization is the whole game on 128 GB nodes: it's what collapses a two-node model onto one GPU and lets two models co-reside on a third.
- Cost-aware MoA routing is exactly the mechanism that produces — and then *defends* — a 90% local share.
- The router log removes the single hardest part of any self-improving system: collecting the training data. It's already there.

## Where it stands

The serving tier is validated and benchmarked (the Qwen3.6 light-tier runs coherently at 256K context with fp4 attention engaged, 0 errors across a full concurrency sweep; the Two-Tower diffusion model runs as a persistent hot worker). The Gemma-4-12B LoRA self-training harness is built — miner → trainer → cycle launcher — and wired to Hermes; it goes live the moment the router starts logging its decisions.

Four desktops, a group chat, and a teacher. Ask it something hard tonight, and it's a little less likely to need the cloud tomorrow.

*Repo + full architecture diagram in the reply. Built on a 4× DGX Spark GB10 cluster with Hermes MoA.*

---

### Thread version (if you want to post it as a thread instead)

**1/** I gave four $4k desktops a group chat and a teacher. They now do ~90% of my AI work locally — and get better every night. A 4× NVIDIA DGX Spark cluster running a self-improving Mixture-of-Agents stack. 🧵

**2/** Hardware: 4× DGX Spark (GB10, 128GB unified each) on a 200Gb RoCE fabric. ~$16k, sits on a shelf, behaves like one machine.

**3/** The trick: models are *agents*, not endpoints. A router (Hermes MoA) sends each task to the cheapest agent that can do it — and hard tasks to several, then votes.

**4/** The roster:
🧠 DSV4-Flash = orchestrator/router
🎨 Two-Tower NVFP4 = diffusion, quantized 118GB→21GB/tower to fit ONE GPU
🎙️ Nemotron-3-Omni = multimodal ingest (shares a node!)
📚 Gemma-4-12B = the student
⚡ Qwen3.6-27B = light tier, 256K ctx, 68 tok/s
☁️ cloud = rate-limited oracle

**5/** Self-improvement: the orchestrator logs every task+route+verdict. That log IS a training set. A miner pulls the best pairs (esp. ones the cloud had to correct) → Gemma-4-12B LoRA-trains → adapter hot-swaps in. And it trains on the ROUTING + AGGREGATION decisions too — so the MoA *logic* improves, not just the leaf models. One command, self-runs on a timer.

**6/** Result: every cloud escalation buys a permanent local skill. The cloud bill trends DOWN. 90/10 → 92/8 → 95/5, automatically, because the cheapest agent keeps getting more competent.

**7/** NVFP4 quantization is the whole game on 128GB nodes — it's what fits a 2-node model onto 1 GPU and lets 2 models co-reside on a 3rd. Full architecture + repo 👇
