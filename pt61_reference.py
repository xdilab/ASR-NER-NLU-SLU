#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, math, os, re, shutil, time, unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SRC = HERE / "source"

# Reuse the exact PT57 Dynamic-k implementation packaged with PT60.
spec = importlib.util.spec_from_file_location("pt57g", SRC / "02_pt57g_asr_specialized_train.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
D = T.D
B = T.B

TRAIN_CSV = HERE / "data" / "train281_asr_target_clean_pt57_v2.csv"
DEV_CSV = HERE / "data" / "dev61_asr_target_clean_pt57_v2.csv"

# Accepted PT57 winner and its train-only routing budget.
PT57 = Path("/data/smgreen1/voxtral_moe_v1/pt57_final_asr_winner_v1_0/GLOBAL_BEST/moe_trainable_state.pt")
BUDGET = Path("/data/smgreen1/voxtral_moe_v1/pt57g_asr_specialized_dynamic_k_recovery_v1_0/PRIMARY_LR5E5/INITIAL_BUDGET_CERT.json")

EXPECTED_SHA = "d44f3708e922f1a6631327c0b24f2e2321b68b1efdc6724752effa477a4e35b6"
EXPECTED_DEV61_WER = 0.259316351905044
EXPECTED_CRIT_ACC = 0.8444444444444444

OUTROOT = Path("/data/smgreen1/voxtral_moe_v1/pt61_pt57_hardmine_critical_expert_recovery_v1_0")

HIGH_VALUE_SLOTS = [
    "ann__slot__full_callsign",
    "ann__slot__flight_number",
    "ann__slot__runway_number",
    "ann__slot__approach_runway",
    "ann__slot__target_altitude",
    "ann__slot__flight_level",
    "ann__slot__target_heading",
    "ann__slot__transfer_frequency",
    "ann__slot__taxiway",
    "ann__slot__planned_route",
    "ann__slot__departure_route",
    "ann__slot__arrival_route",
    "ann__slot__clearance_limit",
    "ann__slot__waypoint_fix",
    "ann__slot__surface_route",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def present(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s not in ("", "nan", "none", "null", "[]", "{}")

def fval(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def build_hard_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    TRAIN-only hard-mining weights.
    No DEV61 errors, predictions, or labels are used to create these weights.
    Every TRAIN281 row is still seen once per epoch; difficult rows simply
    contribute a stronger gradient.
    """
    out = df.copy()
    raws, reasons, slot_counts = [], [], []
    for _, r in out.iterrows():
        raw = 1.0
        rs = []

        role = str(r.get("role_3", ""))
        if role == "Pilot":
            raw += 0.35
            rs.append("pilot")
        elif role not in ("Controller", "Pilot"):
            raw += 0.15
            rs.append("role_other")

        if str(r.get("alignment_method", "")) == "unique_speaker_role":
            raw += 0.20
            rs.append("target_speaker_isolation")

        crit_frac = max(0.0, min(1.0, fval(r.get("critical_word_fraction", 0.0))))
        if crit_frac > 0:
            raw += 0.80 * crit_frac
            rs.append(f"critical_fraction={crit_frac:.3f}")

        nslots = sum(present(r.get(c)) for c in HIGH_VALUE_SLOTS)
        if nslots:
            raw += min(0.60, 0.06 * nslots)
            rs.append(f"high_value_slots={nslots}")

        if fval(r.get("ann__contains_hesitation", 0)):
            raw += 0.10
            rs.append("hesitation")
        if fval(r.get("ann__contains_noise_marker", 0)):
            raw += 0.08
            rs.append("noise_marker")
        if fval(r.get("ann__multiple_speakers_present", 0)):
            raw += 0.15
            rs.append("multiple_speakers")
        if fval(r.get("ann__contains_unclear_unknown", 0)):
            raw += 0.18
            rs.append("unclear_unknown")

        raw = min(2.50, raw)
        raws.append(raw)
        reasons.append("|".join(rs) if rs else "ordinary")
        slot_counts.append(nslots)

    raw = np.asarray(raws, dtype=np.float64)

    # Deliberately emphasize the difficult tail, then normalize mean weight to 1.
    # This preserves the approximate effective learning-rate scale.
    weights = np.square(raw)
    weights = weights / weights.mean()

    out["pt61_raw_hard_score"] = raw
    out["pt61_train_weight"] = weights
    out["pt61_high_value_slot_count"] = slot_counts
    out["pt61_weight_reasons"] = reasons
    return out

def load_pt57_state(model):
    if not PT57.exists():
        raise FileNotFoundError(f"Accepted PT57 checkpoint missing: {PT57}")
    actual = sha256(PT57)
    print("PT57_SHA256:", actual)
    if actual != EXPECTED_SHA:
        raise RuntimeError(f"PT57 SHA mismatch: {actual} != {EXPECTED_SHA}")

    state = torch.load(PT57, map_location="cpu")
    named = dict(model.named_parameters())
    bad = []
    loaded = 0
    for n, t in state.items():
        if n not in named or tuple(named[n].shape) != tuple(t.shape):
            bad.append(n)
            continue
        named[n].data.copy_(t.to(device=named[n].device, dtype=named[n].dtype))
        loaded += 1
    print("PT57_STATE_TENSORS:", len(state))
    print("PT57_LOADED_TENSORS:", loaded)
    if bad or loaded != len(state):
        raise RuntimeError(f"PT57 load failure; bad={bad[:10]}")
    return actual

def set_accepted_policy():
    if not BUDGET.exists():
        raise FileNotFoundError(f"PT57 routing budget missing: {BUDGET}")
    b = json.loads(BUDGET.read_text())
    if not b.get("cert_pass"):
        raise RuntimeError("Accepted PT57 routing budget certificate is not PASS.")
    D.TAU1 = float(b["tau1"])
    D.TAU2 = float(b["tau2"])
    D.ALPHA_U = 1.0
    D.BETA_C = 0.0
    D.DYNAMIC_ENABLED = True
    D.CURRENT_CRIT_TARGET = None
    print("ACCEPTED_TAU1:", D.TAU1)
    print("ACCEPTED_TAU2:", D.TAU2)

def set_experts_only_trainable(model):
    experts, routers, crit = [], [], []
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.endswith(".moe_A") or n.endswith(".moe_B"):
            p.requires_grad = True
            experts.append(p)
        elif n.startswith("moe_router_bank."):
            routers.append(p)
        elif n.startswith("moe_criticality_bank."):
            crit.append(p)
    print("TRAINABLE_EXPERT_PARAMS:", sum(p.numel() for p in experts))
    print("FROZEN_ROUTER_PARAMS:", sum(p.numel() for p in routers))
    print("FROZEN_CRITICALITY_PARAMS:", sum(p.numel() for p in crit))
    if not experts:
        raise RuntimeError("No expert parameters found.")
    if any(p.requires_grad for p in routers + crit):
        raise RuntimeError("Router/criticality freeze failed.")
    return experts, routers, crit

def snapshot(ps):
    return [p.detach().float().cpu().clone() for p in ps]

def delta_l2(before, ps):
    total = 0.0
    for a, p in zip(before, ps):
        d = p.detach().float().cpu() - a
        total += float((d * d).sum())
    return math.sqrt(total)

def norm_words(s):
    s = unicodedata.normalize("NFKC", "" if s is None else str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().split()

def ref_alignment_status(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = "D"
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = "I"
    pri = {"S":0, "D":1, "I":2}
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
                bt[i][j] = "="
            else:
                c = [
                    (dp[i-1][j-1] + 1, "S"),
                    (dp[i-1][j] + 1, "D"),
                    (dp[i][j-1] + 1, "I"),
                ]
                dp[i][j], bt[i][j] = min(c, key=lambda z:(z[0], pri[z[1]]))
    st = ["?"] * n
    i, j = n, m
    while i or j:
        op = bt[i][j]
        if op in ("=", "S"):
            st[i-1] = op; i -= 1; j -= 1
        elif op == "D":
            st[i-1] = "D"; i -= 1
        else:
            j -= 1
    return r, st

def parse_json(v, default):
    try:
        if pd.isna(v):
            return default
        return json.loads(str(v))
    except Exception:
        return default

@torch.no_grad()
def infer_one(model, processor, audio_path, device):
    model.eval()
    x = B.prompt_inputs(processor, audio_path, device)
    plen = x["input_ids"].shape[1]
    t0 = time.time()
    out = model.generate(
        **x,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.0,
    )
    sec = time.time() - t0
    txt = processor.batch_decode(out[:, plen:], skip_special_tokens=True)[0]
    return B.clean(txt), sec, int(out.shape[1] - plen)

@torch.no_grad()
def evaluate(model, processor, df, device, label, out_csv):
    model.eval()
    D.DYNAMIC_ENABLED = True
    D.CURRENT_CRIT_TARGET = None
    D.reset_stats(model)

    rows = []
    refs, hyps = [], []
    S = De = I = 0
    crit_ok = crit_total = non_ok = non_total = 0
    slot = defaultdict(lambda: {"correct":0, "total":0})

    print(f"\n=== {label} ===")
    for ii, (_, r) in enumerate(df.iterrows(), 1):
        g = B.clean_transcript(r["asr_target_clean"])
        p, sec, toks = infer_one(model, processor, r["resolved_audio_path"], device)
        s, d, ins = T.align_counts(g, p)
        S += s; De += d; I += ins
        refs.append(g); hyps.append(p)

        rw, st = ref_alignment_status(g, p)
        scores = parse_json(r.get("critical_word_scores_json"), [0.0]*len(rw))
        scores = (scores + [0.0]*len(rw))[:len(rw)]
        for idx, sc in enumerate(scores):
            ok = idx < len(st) and st[idx] == "="
            if float(sc) > 0:
                crit_total += 1; crit_ok += int(ok)
            else:
                non_total += 1; non_ok += int(ok)

        spans = parse_json(r.get("critical_spans_json"), [])
        for sp in spans:
            a = int(sp.get("start_word", 0))
            b = int(sp.get("end_word_exclusive", a))
            key = str(sp.get("slot", "UNKNOWN"))
            ok = all(0 <= k < len(st) and st[k] == "=" for k in range(a, b))
            slot[key]["total"] += 1
            slot[key]["correct"] += int(ok)

        rows.append({
            "audio_name": r["audio_name"],
            "role_3": r.get("role_3"),
            "alignment_method": r.get("alignment_method"),
            "reference_transcript": g,
            "predicted_transcript": p,
            "row_wer_ehsan_style": B.row_wer(g, p),
            "row_cer": B.row_cer(g, p),
            "token_f1": B.token_f1(g, p),
            "runtime_sec": sec,
            "generated_tokens": toks,
        })
        if ii <= 3 or ii % 20 == 0 or ii == len(df):
            mean = float(np.mean([x["row_wer_ehsan_style"] for x in rows]))
            print(f"[{ii:02d}/{len(df)}] WER={rows[-1]['row_wer_ehsan_style']:.3f} mean={mean:.6f}")

    z = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(out_csv, index=False)

    slot_rows = [
        {
            "slot": k,
            "correct_spans": v["correct"],
            "total_spans": v["total"],
            "span_preservation_accuracy": v["correct"] / max(1, v["total"]),
        }
        for k, v in slot.items()
    ]
    slot_df = pd.DataFrame(slot_rows)
    if len(slot_df):
        slot_df = slot_df.sort_values(["span_preservation_accuracy", "total_spans"], ascending=[True, False])

    metrics = {
        "n": len(z),
        "mean_row_WER_Ehsan_style": float(z["row_wer_ehsan_style"].mean()),
        "corpus_WER": float(B.corpus_wer(refs, hyps)),
        "mean_row_CER": float(z["row_cer"].mean()),
        "mean_TokenF1": float(z["token_f1"].mean()),
        "substitutions": int(S),
        "deletions": int(De),
        "insertions": int(I),
        "critical_word_exact_accuracy": crit_ok / max(1, crit_total),
        "critical_words_correct": int(crit_ok),
        "critical_words_total": int(crit_total),
        "noncritical_word_exact_accuracy": non_ok / max(1, non_total),
        "role_wer": {
            str(k): float(g["row_wer_ehsan_style"].mean())
            for k, g in z.groupby("role_3", dropna=False)
        },
        **D.route_summary(model),
    }
    print("METRICS:", json.dumps(metrics, indent=2))
    return metrics, slot_df

def save_state(model, outdir, meta):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state = {}
    for n, p in model.named_parameters():
        if n.endswith(".moe_A") or n.endswith(".moe_B") or n.startswith("moe_router_bank."):
            state[n] = p.detach().cpu()
    if len(state) != 624:
        raise RuntimeError(f"Expected 624 PT57-compatible tensors, got {len(state)}")
    torch.save(state, outdir / "moe_trainable_state.pt")
    (outdir / "BEST_META.json").write_text(json.dumps(meta, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--expert-lr", type=float, default=1.0e-5)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--calibration-rows", type=int, default=32)
    ap.add_argument("--cert-rows", type=int, default=16)
    args = ap.parse_args()

    OUTROOT.mkdir(parents=True, exist_ok=True)
    log_config = vars(args).copy()
    log_config.update({
        "experiment": "PT61",
        "initialization": "accepted PT57 Rank-16 Dynamic-k winner",
        "trainable": "experts only",
        "router": "frozen",
        "hard_mining": "TRAIN-only metadata weights; no DEV-derived weights",
        "official175_used": False,
    })
    (OUTROOT / "RUN_CONFIG.json").write_text(json.dumps(log_config, indent=2))

    train = pd.read_csv(TRAIN_CSV, low_memory=False)
    dev = pd.read_csv(DEV_CSV, low_memory=False)
    if len(train) != 281 or len(dev) != 61:
        raise RuntimeError(f"Unexpected cohort sizes train={len(train)} dev={len(dev)}")

    weighted = build_hard_weights(train)
    weighted.to_csv(OUTROOT / "TRAIN281_HARD_WEIGHTS.csv", index=False)
    print("WEIGHT_STATS:")
    print(weighted["pt61_train_weight"].describe().to_string())
    print("TOP_HARD_ROWS:")
    print(weighted.sort_values("pt61_train_weight", ascending=False)[
        ["audio_name","role_3","alignment_method","critical_word_fraction",
         "pt61_high_value_slot_count","pt61_train_weight","pt61_weight_reasons"]
    ].head(20).to_string(index=False))

    print("\n=== LOAD ACCEPTED PT57 ===")
    model, processor, device = B.load_base(0)
    D.install_dynamic_moe(model)
    pt57_sha = load_pt57_state(model)
    set_accepted_policy()

    # Strict pre-training parity under the exact PT60 greedy decoder.
    pre, pre_slots = evaluate(model, processor, dev, device, "PT61 PRETRAIN PARITY", OUTROOT / "DEV61_PRETRAIN.csv")
    pre_slots.to_csv(OUTROOT / "DEV61_PRETRAIN_SLOT_SPANS.csv", index=False)
    if abs(pre["mean_row_WER_Ehsan_style"] - EXPECTED_DEV61_WER) > 1e-9:
        raise RuntimeError(
            f"PT57 baseline parity failed: {pre['mean_row_WER_Ehsan_style']} != {EXPECTED_DEV61_WER}"
        )
    if abs(pre["critical_word_exact_accuracy"] - EXPECTED_CRIT_ACC) > 1e-9:
        raise RuntimeError(
            f"PT57 critical-word parity failed: {pre['critical_word_exact_accuracy']} != {EXPECTED_CRIT_ACC}"
        )
    print("BASELINE_PARITY: PASS")

    experts, routers, crit = set_experts_only_trainable(model)
    expert_before = snapshot(experts)
    router_before = snapshot(routers)

    opt = AdamW(experts, lr=args.expert_lr, weight_decay=args.weight_decay)

    best_wer = EXPECTED_DEV61_WER
    best_crit = EXPECTED_CRIT_ACC
    best_epoch = 0
    stale = 0
    history = []
    grad_nonzero_steps = 0
    t0 = time.time()

    # Fixed predeclared hard-mining strategy; DEV61 is used only for epoch selection.
    for ep in range(1, args.epochs + 1):
        print(f"\n=== PT61 HARD-MINED EXPERT TRAIN EPOCH {ep} ===")
        D.DYNAMIC_ENABLED = True
        D.CURRENT_CRIT_TARGET = None
        model.train()
        order = weighted.sample(frac=1.0, random_state=6100 + ep).reset_index(drop=True)

        opt.zero_grad(set_to_none=True)
        task_losses, weighted_losses = [], []
        skipped = 0

        pb = tqdm(order.iterrows(), total=len(order), desc=f"PT61 ep{ep}", dynamic_ncols=True, mininterval=5)
        for step, (_, r) in enumerate(pb, 1):
            batch = B.make_train_batch(
                processor,
                r["resolved_audio_path"],
                B.clean_transcript(r["asr_target_clean"]),
                device,
            )
            out = model(**batch, use_cache=False, return_dict=True)
            task = out.loss
            w = float(r["pt61_train_weight"])
            loss = task * w

            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                del batch, out, task, loss
                continue

            (loss / args.grad_accum).backward()

            if any(p.grad is not None and p.grad.detach().abs().sum().item() > 0 for p in experts):
                grad_nonzero_steps += 1

            if step % args.grad_accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(experts, args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

            task_losses.append(float(task.detach()))
            weighted_losses.append(float(loss.detach()))
            pb.set_postfix(task=f"{task_losses[-1]:.3f}", w=f"{w:.2f}", loss=f"{weighted_losses[-1]:.3f}")
            del batch, out, task, loss

        pb.close()
        gc.collect()
        torch.cuda.empty_cache()

        # Experts changed the hidden states, so restore the Dynamic-k compute budget
        # with TRAIN-only calibration. Router weights remain frozen.
        budget_ep = T.exact_budget_calibrate(
            model, processor, weighted, device,
            cal_rows=args.calibration_rows,
            cert_rows=args.cert_rows,
            iters=1,
            seed_base=6100 + ep * 10,
        )
        (OUTROOT / f"TRAIN_ONLY_BUDGET_CERT_EPOCH_{ep}.json").write_text(json.dumps(budget_ep, indent=2))
        if not budget_ep["cert_pass"]:
            print("WARNING: TRAIN-only Dynamic-k budget certificate failed for this epoch.")

        metrics, slot_df = evaluate(
            model, processor, dev, device,
            f"PT61 DEV61 EPOCH {ep}",
            OUTROOT / f"DEV61_EPOCH_{ep}.csv",
        )
        slot_df.to_csv(OUTROOT / f"DEV61_SLOT_SPANS_EPOCH_{ep}.csv", index=False)

        rec = {
            "epoch": ep,
            "train_task_loss": float(np.mean(task_losses)) if task_losses else None,
            "train_weighted_loss": float(np.mean(weighted_losses)) if weighted_losses else None,
            "skipped": skipped,
            "budget_cert_pass": bool(budget_ep["cert_pass"]),
            **metrics,
        }
        history.append(rec)
        pd.DataFrame(history).to_csv(OUTROOT / "HISTORY.csv", index=False)

        # Primary selection = WER. Critical-word accuracy breaks near-exact ties.
        valid = bool(budget_ep["cert_pass"]) and 1.80 <= metrics["average_routed_k"] <= 1.98
        improved = valid and (
            metrics["mean_row_WER_Ehsan_style"] < best_wer - 1e-12
            or (
                abs(metrics["mean_row_WER_Ehsan_style"] - best_wer) <= 1e-12
                and metrics["critical_word_exact_accuracy"] > best_crit + 1e-12
            )
        )

        if improved:
            best_wer = metrics["mean_row_WER_Ehsan_style"]
            best_crit = metrics["critical_word_exact_accuracy"]
            best_epoch = ep
            stale = 0
            meta = {
                "experiment": "PT61",
                "best_epoch": ep,
                "best_metrics": metrics,
                "baseline_wer": EXPECTED_DEV61_WER,
                "baseline_critical_word_exact_accuracy": EXPECTED_CRIT_ACC,
                "absolute_wer_gain": EXPECTED_DEV61_WER - best_wer,
                "relative_wer_reduction": (EXPECTED_DEV61_WER - best_wer) / EXPECTED_DEV61_WER,
                "critical_word_accuracy_gain": best_crit - EXPECTED_CRIT_ACC,
                "trainable": "Rank-16 experts only",
                "router": "frozen; thresholds recalibrated using TRAIN only after each epoch",
                "hard_mining": "TRAIN-only metadata weighting",
                "official175_used": False,
                "model_selection_set": "DEV61",
                "pt57_sha256": pt57_sha,
            }
            save_state(model, OUTROOT / "GLOBAL_BEST", meta)
            shutil.copy2(OUTROOT / f"DEV61_EPOCH_{ep}.csv", OUTROOT / "GLOBAL_BEST" / "DEV61_BEST.csv")
            shutil.copy2(OUTROOT / f"DEV61_SLOT_SPANS_EPOCH_{ep}.csv", OUTROOT / "GLOBAL_BEST" / "DEV61_SLOT_SPANS_BEST.csv")
            print(f"*** NEW PT61 BEST: WER={best_wer:.9f} CRIT_ACC={best_crit:.6f} ep={ep} ***")
        else:
            stale += 1
            print(f"NO ACCEPTED IMPROVEMENT {stale}/{args.patience}; current={metrics['mean_row_WER_Ehsan_style']:.9f} best={best_wer:.9f}")

        if stale >= args.patience:
            print("EARLY STOP:", ep)
            break

    expert_delta = delta_l2(expert_before, experts)
    router_delta = delta_l2(router_before, routers)

    cert = {
        "status": "PASS",
        "experiment": "PT61_PT57_HARDMINE_CRITICAL_EXPERT_RECOVERY_v1_0",
        "pt57_sha256": pt57_sha,
        "official175_used": False,
        "train_rows": len(weighted),
        "dev_rows": len(dev),
        "baseline_parity": True,
        "baseline_wer": EXPECTED_DEV61_WER,
        "baseline_critical_word_exact_accuracy": EXPECTED_CRIT_ACC,
        "best_epoch": best_epoch,
        "best_dev61_wer": best_wer,
        "best_critical_word_exact_accuracy": best_crit,
        "absolute_wer_gain": EXPECTED_DEV61_WER - best_wer,
        "relative_wer_reduction": (EXPECTED_DEV61_WER - best_wer) / EXPECTED_DEV61_WER,
        "critical_word_accuracy_gain": best_crit - EXPECTED_CRIT_ACC,
        "beats_pt57": bool(best_epoch > 0 and best_wer < EXPECTED_DEV61_WER - 1e-12),
        "expert_gradient_nonzero_steps": grad_nonzero_steps,
        "expert_delta_l2": expert_delta,
        "router_delta_l2_should_be_zero": router_delta,
        "training_runtime_sec": time.time() - t0,
        "selection_note": "Single predeclared TRAIN-only hard-mining strategy. DEV61 selects epoch only. official175 untouched.",
    }
    (OUTROOT / "PT61_CERTIFICATE.json").write_text(json.dumps(cert, indent=2))
    print("\nPT61_CERTIFICATE:")
    print(json.dumps(cert, indent=2))
    print("DONE:", OUTROOT)

if __name__ == "__main__":
    main()
