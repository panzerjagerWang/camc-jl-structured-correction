"""Fit a 4B LoRA adapter using authorized local inputs; see README.md."""
from pathlib import Path
import argparse, hashlib, json, math, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You correct Japanese grammar. Replace only the marked error span. Return only the replacement text, with no explanation."

def encoded(tok, row):
    text, start, end = row["text"], row["gold_span_start"], row["gold_span_end"]
    revision = row["gold_revision"]
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
        raise ValueError("Invalid character offsets")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Nonempty literal revision required")
    marked = text[:start] + "<error>" + text[start:end] + "</error>" + text[end:]
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Error type: {row['error_type']}\nSentence: {marked}"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    response = tok.encode(revision, add_special_tokens=False)
    assert tok.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False) == revision
    assert tok.eos_token_id not in response
    response.append(tok.eos_token_id)
    assert len(prompt) + len(response) <= 512, "Input exceeds frozen 512-token fitting contract"
    return dict(ids=prompt+response, labels=[-100]*len(prompt)+response,
                prompt_length=len(prompt), response_length=len(response))

def collate(tok, rows):
    width = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["ids"])
        ids[i, :n] = torch.tensor(row["ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        attention[i, :n] = 1
        assert labels[i, :row["prompt_length"]].eq(-100).all()
        assert int(labels[i].ne(-100).sum()) == row["response_length"]
    return {k: v.cuda() for k, v in
            [("input_ids", ids), ("attention_mask", attention), ("labels", labels)]}

def rate(step, total, peak):
    warm = max(1, int(.05 * total))
    return peak * (step/warm if step <= warm else
                   .1 + .45*(1+math.cos(math.pi*(step-warm)/(total-warm))))

def update(model, tok, optimizer, rows, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(r["response_length"] for r in rows)
    weighted = 0.
    for first in range(0, len(rows), 2):
        part = rows[first:first+2]
        loss = model(**collate(tok, part), use_cache=False).loss
        assert torch.isfinite(loss)
        weight = sum(r["response_length"] for r in part)/denominator
        (loss*weight).backward()
        weighted += float(loss.detach())*weight
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    assert torch.isfinite(norm)
    optimizer.step()
    return weighted, float(norm)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["auxiliary", "source"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT/"model/base")
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.stage == "source" and not args.initial_adapter:
        parser.error("Source adaptation requires the auxiliary step-1000 adapter")
    if args.stage == "auxiliary" and args.seed != 20260928:
        parser.error("The fixed auxiliary sampler and initialization seed is 20260928")
    if args.output.exists():
        parser.error("Use a new output directory; existing checkpoints are never overwritten")
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    raw = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    assert len(raw) == (59767 if args.stage == "auxiliary" else 200), "Unexpected fitting row count"
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    assert tok.pad_token_id is not None and tok.eos_token_id is not None
    rows = [encoded(tok, r) for r in raw]
    del raw
    base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    assert sum(p.numel() for p in base.parameters()) == 4022468096
    base.config.use_cache = False
    if args.initial_adapter:
        model = PeftModel.from_pretrained(base, args.initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=8,
            lora_alpha=16, lora_dropout=.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 16515072
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    assert all(p.dtype == (torch.float32 if "lora_" in n else torch.bfloat16)
               for n, p in model.named_parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.0002 if args.stage == "auxiliary" else .0001, weight_decay=.01)
    assert not optimizer.state
    rng = np.random.default_rng(args.seed)
    if args.stage == "auxiliary":
        order = rng.permutation(59767)[:16000].astype("<i8")
        assert hashlib.sha256(order.tobytes()).hexdigest() == "22dc54c104d82643973ee42c6c030cddc775eb1eefa31c7f945b66be7809a0f3"
        batches = [order[first:first+16] for first in range(0, 16000, 16)]
        total, peak = 1000, .0002
    else:
        batches = []
        for epoch in range(10):
            order = rng.permutation(200)
            batches.extend(order[first:first+16] for first in range(0, 200, 16))
        total, peak = 390, .0001
    logs = []
    args.output.mkdir(parents=True, exist_ok=False)
    for step, indices in enumerate(batches, 1):
        loss, norm = update(model, tok, optimizer, [rows[int(i)] for i in indices], rate(step, total, peak))
        logs.append(dict(step=step, examples=len(indices), loss=loss, gradient_norm=norm,
                         learning_rate=rate(step, total, peak)))
        if step % 25 == 0:
            print(args.stage, step, loss, flush=True)
    model.save_pretrained(args.output, safe_serialization=True)
    config_path = args.output/"adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "Qwen/Qwen3-4B-Instruct-2507"
    config["revision"] = "cdbee75f17c01a7cc42f958dc650907174af0554"
    config_path.write_text(json.dumps(config, indent=2)+"\n")
    with args.input.open("rb") as f:
        input_hash = hashlib.file_digest(f, "sha256").hexdigest()
    (args.output/"training_log.json").write_text(json.dumps(
        dict(stage=args.stage, seed=args.seed, input_sha256=input_hash, logs=logs,
             historical_test_used=False, bitwise_historical_reproduction_asserted=False), indent=2)+"\n")

if __name__ == "__main__":
    main()
