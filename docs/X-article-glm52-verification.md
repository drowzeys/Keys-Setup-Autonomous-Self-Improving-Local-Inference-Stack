# I rebuilt @tonyd2wild's GLM-5.2 200K recipe on a *second* 4× DGX Spark cluster — and it reproduces.

A 355B-class reasoning model, unpruned, **200K context, running on four $4,000 desktops.** Someone
published the full recipe. I built it from scratch on my own cluster to see if it holds. **It holds.**

---

**The claim:** GLM-5.2 (QuantTrio Int4-Int8Mix, all 256 experts) served **TP=4 across four NVIDIA DGX
Sparks (GB10, 128 GB unified each)** at **200K context**, with in-checkpoint MTP speculative decode and
fp8 sparse-MLA KV. No datacenter, no H100s — four desktops on a 200 G RoCE switch.

**The test:** I took the recipe cold — built the vLLM image, baked the sparse-MLA + b12x mods and the
indexer patch, staged 378 GB of weights + NCCL + kernels to all four nodes, and launched. Independent
cluster, independent hardware.

## It reproduces — eval 4/4, zero crashes

| Test | Result |
|---|---|
| Correctness eval | **4 / 4** (math, logic, code, factual) |
| Concurrency C1→C6 | **0 crashes** — the indexer patch works exactly as documented |
| MTP mean acceptance | **3.0–3.2** (recipe says ~3.2) ✅ |
| 127K passkey retrieval | ✅ correct at depth-75% |

## Throughput

| Metric | Mine | Reference | 
|---|--:|--:|
| Decode, C6 aggregate | **55.3 tok/s** | 60.5 |
| Decode, single-stream (thinking-off) | **34.3 tok/s** | 28.8 |
| Prefill @64K | **996 tok/s** | ~700 |

Single-stream, apples-to-apples, I actually **beat the reference (34.3 vs 28.8)**.

## The one subtlety worth sharing

My *first* numbers looked ~30% low. The cause wasn't the engine, the mods, or the fabric — those all
matched (MTP acceptance was dead-on 3.2). It was **reasoning mode**: GLM-5.2 *thinks* before it answers,
and that thinking burns ~40% of wall-time. Turn thinking off (or use low-depth prompts, like the
reference bench) → the "gap" vanishes and you land **above** 28.8. Decode tok/s is dominated by two
things only: **reasoning on/off**, and **MTP acceptance** (how predictable the output is — 17 tok/s for
creative prose, 34 for a predictable list). No config defect anywhere.

## Bonus finding

Two red herrings I chased so you don't have to:
- **MTU 9000 (jumbo frames)** helps *prefill* (large transfers) but does **nothing** for decode — decode
  all-reduces are a few KB, latency-bound, MTU-insensitive. Set it for TTFT, not tok/s.
- One **build fix**: the image's PR-35568 patch step hard-fails against the pinned commit (already
  merged upstream) — make that `git apply` non-fatal and the build sails through.

Full independent verification writeup + build notes in the fork. **Enormous credit to @tonyd2wild** for
publishing a recipe complete enough to reproduce, and to CosmicRaisins / ciprianveg / eugr for the
sparse-MLA kernels and build harness. This is what open-weights + open-recipe looks like.

*Four desktops. A frontier-class reasoning model. 200K context. Reproduced.*

---

### Thread version

**1/** Someone published the full recipe to run GLM-5.2 (355B-class, unpruned, 200K ctx) on 4× NVIDIA
DGX Spark desktops. I rebuilt it from scratch on my own cluster to see if it holds. It does. 🧵

**2/** Independent 4× DGX Spark (GB10, 128GB each) on a MikroTik 200G RoCE switch. Built the vLLM image,
baked the sparse-MLA + indexer-overhang patches, staged 378GB of weights to all 4 nodes, launched TP=4.

**3/** Result: eval **4/4**, **zero crashes C1→C6** (the indexer patch works), MTP acceptance **3.0–3.2**
(recipe says 3.2), 127K passkey retrieval ✓. Faithful reproduction.

**4/** Throughput: C6 **55 tok/s**, prefill **996 tok/s @64K**, and single-stream apples-to-apples
**34.3 tok/s — above the reference 28.8.**

**5/** My first numbers looked low. Cause: GLM-5.2 *reasons* before answering (~40% of wall-time). Turn
thinking off → gap vanishes, land above 28.8. Decode speed = reasoning on/off × MTP acceptance. No config
defect.

**6/** Two red herrings: MTU-9000 jumbo frames help *prefill* not decode (decode all-reduces are tiny).
And the image's PR-35568 patch fails against the pinned commit (already merged) — make it non-fatal.

**7/** Huge credit to @tonyd2wild for a recipe complete enough to reproduce, + CosmicRaisins/ciprianveg/
eugr for the kernels. Full verification writeup 👇 Open weights + open recipe = this.
