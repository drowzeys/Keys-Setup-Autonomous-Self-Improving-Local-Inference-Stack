#!/usr/bin/env python3
"""LoRA-fine-tune Gemma-4-12B-it on mined Hermes signal (train_pairs.jsonl).

Gemma is the STUDENT of the self-improving MoA: it absorbs (a) specialist answers,
(b) routing decisions and (c) aggregation preferences, so training it improves the
WHOLE MoA logic — not just leaf answers. Sample weights up-rank cloud-gold + routing pairs.

Runs in a container that has torch + peft + trl (base python here has neither):
  pip install -q peft trl transformers accelerate datasets bitsandbytes
Then: python3 gemma_lora_train.py --data train_pairs.jsonl --out adapters/gemma-moa-$(date)
Produces a LoRA adapter dir (hot-swappable) + writes adapter_card.json.
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models-gemma4-12b-it"))
    ap.add_argument("--data", default="train_pairs.jsonl")
    ap.add_argument("--out", default="adapters/gemma-moa")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(a.data)]
    if not rows:
        raise SystemExit("no training pairs — run mine_signal.py first (need accumulated log)")
    tok = AutoTokenizer.from_pretrained(a.model)

    def fmt(r):
        # weight -> integer replication (cheap importance weighting)
        text = tok.apply_chat_template(r["messages"], tokenize=False)
        return [{"text": text}] * max(1, round(r.get("weight", 1.0)))
    flat = [x for r in rows for x in fmt(r)]
    ds = Dataset.from_list(flat)
    print(f"[train] {len(rows)} pairs -> {len(flat)} weighted examples")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs,
                    per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    learning_rate=a.lr, max_seq_length=a.max_len, bf16=True,
                    logging_steps=5, save_strategy="epoch", report_to=[])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.out)
    json.dump({"base": a.model, "n_pairs": len(rows), "rank": a.rank,
               "kinds": "specialist+routing+aggregation+task_outcome"},
              open(os.path.join(a.out, "adapter_card.json"), "w"), indent=2)
    print(f"[train] saved adapter -> {a.out}")

if __name__ == "__main__":
    main()
