#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

VOXTRAL_MODEL = "/home/ealam/Boeing W1 /LOCAL_AUDIO_MODELS/Voxtral-Small-24B"
PROMPT_HINT = "You are an aviation expert now transcribe this audio only provide the text transcribe nothing more than that."
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
PRECISION = "4bit"
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
TEST_FRAC = 0.2
SPLIT_SEED = 42
ATCOSIM_METADATA_DEFAULT = "/home/ealam/Boeing W1 /ATCOSIM_PLAIN_ASR_METADATA.csv"
OUTPUT_DIR_DEFAULT = "/home/ealam/Boeing W1 /Gemma_ATC/Gemma12basr/results_voxtral/GITHUB_ATCOSIM"
STAGE_LABEL = "ATCOSIM_VOXTRAL_STAGE1"


def clean(x):
    import re
    return re.sub(r"\s+", " ", "" if x is None else str(x)).strip()


def word_error_rate(gold, pred):
    g, p = gold.split(), pred.split()
    n, m = len(g), len(p)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if g[i - 1] == p[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(1, n)


def char_error_rate(gold, pred):
    g, p = list(gold), list(pred)
    n, m = len(g), len(p)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if g[i - 1] == p[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(1, n)


def token_f1(gold, pred):
    from collections import Counter
    g, p = gold.split(), pred.split()
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    gc, pc = Counter(g), Counter(p)
    overlap = sum((gc & pc).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(gold, pred):
    g, p = gold.split(), pred.split()
    if not g or not p:
        return 0.0
    l = lcs_len(g, p)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(g)
    return 2 * prec * rec / (prec + rec)


def build_cosine_fn():
    try:
        from sentence_transformers import SentenceTransformer
        import torch.nn.functional as F
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        def cosine(gold, pred):
            emb = st_model.encode([gold, pred], convert_to_tensor=True, show_progress_bar=False)
            return F.cosine_similarity(emb[0:1], emb[1:2]).item()

        return cosine
    except Exception:
        from collections import Counter
        import math

        def cosine(gold, pred):
            g, p = Counter(gold.split()), Counter(pred.split())
            if not g or not p:
                return 0.0
            keys = set(g) | set(p)
            dot = sum(g[k] * p[k] for k in keys)
            ng = math.sqrt(sum(v * v for v in g.values()))
            npd = math.sqrt(sum(v * v for v in p.values()))
            return dot / (ng * npd) if ng and npd else 0.0

        return cosine


COSINE_FN = build_cosine_fn()

try:
    import sacrebleu
    HAVE_BLEU = True
except ImportError:
    HAVE_BLEU = False


def make_splits(metadata_csv, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    df = pd.read_csv(metadata_csv, low_memory=False)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return df, train_df, val_df, test_df


def choose_subset(df, n, seed):
    if n <= 0 or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)


def build_conversation(audio_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": PROMPT_HINT},
            ],
        }
    ]


def build_vanilla_model(device):
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(VOXTRAL_MODEL, local_files_only=True)
    if PRECISION == "bf16":
        model = VoxtralForConditionalGeneration.from_pretrained(
            VOXTRAL_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map="balanced", local_files_only=True,
        )
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = VoxtralForConditionalGeneration.from_pretrained(
            VOXTRAL_MODEL, quantization_config=bnb_config, device_map={"": device}, local_files_only=True,
        )
        model.gradient_checkpointing_enable()
    real_device = next(model.parameters()).device
    return model, processor, real_device


def apply_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if PRECISION != "bf16":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_lora_checkpoint(base_device, adapter_dir, is_trainable=True):
    from peft import PeftModel, prepare_model_for_kbit_training
    base_model, processor, real_device = build_vanilla_model(base_device)
    if is_trainable and PRECISION != "bf16":
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=is_trainable)
    return model, processor, real_device


def build_or_resume(device, output_dir):
    output_dir = Path(output_dir)
    latest = output_dir / "LATEST_MIDEPOCH"
    best = output_dir / "GLOBAL_BEST"
    if latest.exists() and any(latest.iterdir()):
        print(f"Resuming from mid-epoch checkpoint: {latest}")
        return load_lora_checkpoint(device, latest, is_trainable=True)
    if best.exists() and any(best.iterdir()):
        print(f"Resuming from best checkpoint: {best}")
        return load_lora_checkpoint(device, best, is_trainable=True)
    model, processor, device = build_vanilla_model(device)
    model = apply_lora(model)
    return model, processor, device


@torch.no_grad()
def run_inference(model, processor, audio_path, device):
    conv = build_conversation(audio_path)
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = inputs.to(device=device)
    prompt_len = inputs["input_ids"].shape[1]
    out_ids = model.generate(
        **inputs, max_new_tokens=256, do_sample=False,
        no_repeat_ngram_size=3, repetition_penalty=1.3,
    )
    text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0]
    return clean(text)


def evaluate_df(model, processor, df, device, max_samples=None, seed=0, log_prefix=""):
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed)
    model.eval()
    rows = []
    pbar = tqdm(df.iterrows(), total=len(df), desc=log_prefix or "eval", mininterval=5.0, dynamic_ncols=True)
    for _, row in pbar:
        gold = clean(row["full_transcript"])
        t0 = time.time()
        pred = run_inference(model, processor, row["audio_path"], device)
        elapsed = time.time() - t0
        duration = row.get("duration_sec", None)
        rtf = elapsed / duration if duration and duration > 0 else None
        wer = word_error_rate(gold, pred)
        rows.append({
            "audio_name": row["audio_name"], "gold": gold, "prediction": pred,
            "WER": wer, "CER": char_error_rate(gold, pred),
            "sacreBLEU": (sacrebleu.sentence_bleu(pred, [gold]).score if HAVE_BLEU else None),
            "Cosine": COSINE_FN(gold, pred), "ROUGE_L": rouge_l(gold, pred),
            "TokenF1": token_f1(gold, pred), "RTF": rtf,
        })
        pbar.set_postfix(wer=f"{wer:.3f}")
    pbar.close()
    d = pd.DataFrame(rows)
    metrics = {
        "WER": d["WER"].mean(), "CER": d["CER"].mean(),
        "sacreBLEU": d["sacreBLEU"].mean() if HAVE_BLEU else None,
        "Cosine": d["Cosine"].mean(), "ROUGE_L": d["ROUGE_L"].mean(),
        "TokenF1": d["TokenF1"].mean(),
        "RTF": d["RTF"].mean() if d["RTF"].notna().any() else None,
    }
    return metrics, d


def compute_train_loss(model, processor, row, device):
    conv = build_conversation(row["audio_path"])
    prompt_inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    prompt_inputs = prompt_inputs.to(device=device)
    gold = clean(row["full_transcript"])
    target_ids = processor.tokenizer(
        gold, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=448,
    ).input_ids.to(device)
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = torch.cat([target_ids, torch.tensor([[eos_id]], device=device)], dim=1)
    prompt_ids = prompt_inputs["input_ids"]
    prompt_mask = prompt_inputs["attention_mask"]
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, torch.ones_like(target_ids)], dim=1)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)
    extra_kwargs = {
        k: v for k, v in prompt_inputs.items()
        if k not in ("input_ids", "attention_mask")
    }
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **extra_kwargs)
    return out.loss


def train_one_epoch(model, processor, optimizer, train_df, device, grad_accum=4, desc="train",
                     mid_ckpt_dir=None, checkpoint_every=500):
    model.train()
    order = train_df.sample(frac=1.0).index
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    losses, skipped = [], 0
    pbar = tqdm(enumerate(order, 1), total=len(order), desc=desc, mininterval=5.0, dynamic_ncols=True)
    for step, idx in pbar:
        row = train_df.loc[idx]
        loss = compute_train_loss(model, processor, row, device)
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss="skip", skipped=skipped)
            continue
        (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(order):
            if all(torch.isfinite(p.grad).all() for p in trainable_params if p.grad is not None):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        pbar.set_postfix(loss=f"{losses[-1]:.3f}", avg=f"{sum(losses) / len(losses):.3f}", skipped=skipped)
        if mid_ckpt_dir and checkpoint_every and step % checkpoint_every == 0:
            Path(mid_ckpt_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(mid_ckpt_dir))
    pbar.close()
    return sum(losses) / max(1, len(losses)), skipped


def run_stage(model, processor, train_df, val_df, epochs, patience, lr, output_dir,
              train_eval_subsample, stage_label, device, grad_accum):
    print(f"[{stage_label}] train={len(train_df)} validation={len(val_df)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    best_wer, epochs_since_improvement, history = float("inf"), 0, []
    mid_ckpt_dir = output_dir / "LATEST_MIDEPOCH"
    for epoch in range(1, epochs + 1):
        train_loss, skipped = train_one_epoch(
            model, processor, optimizer, train_df, device, grad_accum=grad_accum,
            desc=f"{stage_label} ep{epoch} train", mid_ckpt_dir=mid_ckpt_dir, checkpoint_every=500,
        )
        train_metrics, _ = evaluate_df(
            model, processor, train_df, device, max_samples=train_eval_subsample, seed=epoch,
            log_prefix=f"{stage_label} ep{epoch} train-eval",
        )
        val_metrics, val_pred = evaluate_df(
            model, processor, val_df, device, log_prefix=f"{stage_label} ep{epoch} val-eval",
        )
        record = {
            "stage": stage_label, "epoch": epoch, "train_loss": train_loss, "skipped": skipped,
            "train_WER": train_metrics["WER"], "train_CER": train_metrics["CER"],
            "train_sacreBLEU": train_metrics["sacreBLEU"], "train_Cosine": train_metrics["Cosine"],
            "train_ROUGE_L": train_metrics["ROUGE_L"], "train_TokenF1": train_metrics["TokenF1"],
            "train_RTF": train_metrics["RTF"],
            "val_WER": val_metrics["WER"], "val_CER": val_metrics["CER"],
            "val_sacreBLEU": val_metrics["sacreBLEU"], "val_Cosine": val_metrics["Cosine"],
            "val_ROUGE_L": val_metrics["ROUGE_L"], "val_TokenF1": val_metrics["TokenF1"],
            "val_RTF": val_metrics["RTF"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "HISTORY.csv", index=False)
        print(record)
        improved = val_metrics["WER"] < best_wer
        if improved:
            best_wer, epochs_since_improvement = val_metrics["WER"], 0
            best_dir = output_dir / "GLOBAL_BEST"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
        else:
            epochs_since_improvement += 1
        if patience > 0 and epochs_since_improvement >= patience:
            print(f"[{stage_label}] early stopping at epoch {epoch}, best_wer={best_wer:.4f}")
            break
    return output_dir / "GLOBAL_BEST", best_wer


def test_and_export(model, processor, test_df, output_dir, stage_label, device, test_n=0):
    if test_n and test_n < len(test_df):
        test_df = test_df.sample(n=test_n, random_state=99)
    metrics, test_pred = evaluate_df(model, processor, test_df, device, log_prefix=f"{stage_label} test")
    print(f"[{stage_label}] TEST metrics: {metrics}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_pred["Pipeline"] = stage_label
    test_pred.to_csv(output_dir / f"full_test_{stage_label}.csv", index=False)
    best30 = test_pred.sort_values("WER").head(30).copy()
    best30["Prompt"] = PROMPT_HINT
    best30.to_csv(output_dir / f"best30_{stage_label}.csv", index=False)
    return metrics


def run_atcosim(args, device, hp):
    atcosim_full, atcosim_train_df, atcosim_val_df, atcosim_test_df = make_splits(args.atcosim_metadata)
    print(f"[{STAGE_LABEL}] fresh 70/10/20 split -> train={len(atcosim_train_df)} "
          f"val={len(atcosim_val_df)} test={len(atcosim_test_df)} (of {len(atcosim_full)} total, seed={SPLIT_SEED})")
    atcosim_train_df = choose_subset(atcosim_train_df, hp["atcosim_train_n"], 42)
    atcosim_val_df = choose_subset(atcosim_val_df, hp["atcosim_val_n"], 43)
    atcosim_out = Path(args.output_dir) / STAGE_LABEL
    print("Building Voxtral base model + LoRA, resuming if a checkpoint exists...")
    model, processor, device = build_or_resume(device, atcosim_out)
    best_ckpt, best_wer = run_stage(
        model, processor, atcosim_train_df, atcosim_val_df, hp["atcosim_epochs"],
        patience=3, lr=hp["lr"], output_dir=atcosim_out, train_eval_subsample=hp["train_eval_subsample"],
        stage_label=STAGE_LABEL, device=device, grad_accum=hp["grad_accum"],
    )
    del model
    torch.cuda.empty_cache()
    model, processor, device = load_lora_checkpoint(device, best_ckpt, is_trainable=False)
    test_and_export(model, processor, atcosim_test_df, atcosim_out, STAGE_LABEL, device,
                     test_n=hp["atcosim_test_n"])
    print("DONE atcosim. best_wer:", best_wer)


def main():
    global PRECISION
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--atcosim-metadata", default=ATCOSIM_METADATA_DEFAULT)
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = ap.parse_args()
    PRECISION = args.precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "smoke":
        hp = {
            "atcosim_train_n": 6, "atcosim_val_n": 4, "atcosim_epochs": 2, "atcosim_test_n": 6,
            "train_eval_subsample": 4, "lr": 5e-5, "grad_accum": 1,
        }
    else:
        hp = {
            "atcosim_train_n": 8000, "atcosim_val_n": 1000, "atcosim_epochs": 6, "atcosim_test_n": 500,
            "train_eval_subsample": 200, "lr": 5e-5, "grad_accum": 4,
        }
    print("=" * 70)
    print(f"MODE: {args.mode}  EXPERIMENT: {STAGE_LABEL}  DEVICE: {device}")
    print("=" * 70)
    run_atcosim(args, device, hp)


if __name__ == "__main__":
    main()
