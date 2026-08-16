import argparse
import json
import random
import re
import time
from pathlib import Path
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
VOXTRAL_MODEL = '/home/ealam/Boeing W1 /LOCAL_AUDIO_MODELS/Voxtral-Small-24B'
METADATA_CSV = '/home/ealam/Boeing W1 /Gemma_ATC/ATCO2_FINAL_REVIEWED_V4_PACKAGE/ATCO2_FINAL_METADATA_REVIEWED_V4.csv'
DEFAULT_ASR_CHECKPOINT = '/home/ealam/Boeing W1 /Gemma_ATC/Gemma12basr/results_voxtral/ATCO2_VOXTRAL_ONLY/GLOBAL_BEST'
ROLE_INTENT_CHECKPOINT = '/home/ealam/Boeing W1 /Gemma_ATC/Gemma12basr/results_voxtral/GITHUB_OVERSAMPLE/SEQUENTIAL/role_intent/GLOBAL_BEST'
ROLE_COL = 'role_3'
HIGH_INTENT_COL = 'high_level_intent_8'
FINE_INTENT_COL = 'fine_intent_28'
EXCLUDED_SLOT_COLUMNS = {'slot__waypoint_nearby', 'slot__airport_code', 'slot__airport_name', 'slot__controller_unit', 'slot__frequency_context', 'slot__utc_time', 'slot__speaker_identifier', 'slot__speaker_role', 'slot__confidence_or_quality_flag', 'slot__noise_marker', 'slot__nonenglish_marker', 'slot__adc_url', 'slot__coordinates'}
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
PRECISION = '4bit'
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
TEST_FRAC = 0.2
SPLIT_SEED = 42
OVERSAMPLE_MAX_REPEAT = 3
MAX_NEW_TOKENS_DEFAULT = 768
STAGE = 'role_intent_slot'
STRATEGY = 'sequential'
STAGE_LABEL = 'sequential_role_intent_slot'
OUTPUT_DIR = '/home/ealam/Boeing W1 /Gemma_ATC/Gemma12basr/results_voxtral/GITHUB_OVERSAMPLE/SEQUENTIAL/role_intent_slot'

def clean(x):
    return re.sub('\\s+', ' ', '' if x is None else str(x)).strip()

def clean_transcript(x):
    x = '' if x is None else str(x)
    x = re.sub('\\[/?#[^\\]]*\\]', '', x)
    return clean(x)

def word_error_rate(gold, pred):
    (g, p) = (gold.split(), pred.split())
    (n, m) = (len(g), len(p))
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
    (g, p) = (list(gold), list(pred))
    (n, m) = (len(g), len(p))
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
    (g, p) = (gold.split(), pred.split())
    if not g and (not p):
        return 1.0
    if not g or not p:
        return 0.0
    (gc, pc) = (Counter(g), Counter(p))
    overlap = sum((gc & pc).values())
    if overlap == 0:
        return 0.0
    (precision, recall) = (overlap / len(p), overlap / len(g))
    return 2 * precision * recall / (precision + recall)

def lcs_len(a, b):
    (n, m) = (len(a), len(b))
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def rouge_l(gold, pred):
    (g, p) = (gold.split(), pred.split())
    if not g or not p:
        return 0.0
    lcs = lcs_len(g, p)
    (precision, recall) = (lcs / len(p), lcs / len(g))
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    def cosine(gold, pred):
        if not gold or not pred:
            return 0.0
        embs = st_model.encode([gold, pred], convert_to_tensor=True)
        return float(F.cosine_similarity(embs[0:1], embs[1:2]).item())
    COSINE_FN = cosine
except Exception:
    from collections import Counter
    import math

    def cosine(gold, pred):
        (gc, pc) = (Counter(gold.split()), Counter(pred.split()))
        if not gc or not pc:
            return 0.0
        common = set(gc) | set(pc)
        dot = sum((gc.get(w, 0) * pc.get(w, 0) for w in common))
        mag_g = math.sqrt(sum((v * v for v in gc.values())))
        mag_p = math.sqrt(sum((v * v for v in pc.values())))
        return dot / (mag_g * mag_p) if mag_g and mag_p else 0.0
    COSINE_FN = cosine
try:
    import sacrebleu
    HAVE_BLEU = True
except Exception:
    HAVE_BLEU = False

def stage_fields(stage):
    if stage == 'role':
        return ['TRANSCRIPT', 'ROLE']
    if stage == 'role_intent':
        return ['TRANSCRIPT', 'ROLE', 'HIGH_INTENT', 'FINE_INTENT']
    return ['TRANSCRIPT', 'ROLE', 'HIGH_INTENT', 'FINE_INTENT', 'SLOTS']
HIGH_INTENT_CLASSES = ['Command', 'Clearance', 'Confirmation', 'Report', 'Information', 'Request', 'Restriction', 'Other']
FINE_INTENT_CLASSES = ['Flight route', 'Readback', 'Information', 'Status report', 'Transfer', 'Landing', 'Simple respond', 'Report', 'Taxiing', 'Altitude', 'Other', 'Approach', 'Line-up', 'Take-off', 'Acknowledgement', 'Request', 'Confirmation', 'Heading', 'Pushback', 'Contact', 'Speed', 'Restriction']

def build_prompt(stage, slot_columns=None):
    fields = stage_fields(stage)
    lines = ['You are an aviation expert. Listen to this ATC/pilot radio transmission and respond in exactly this format, one field per line, nothing else:']
    for f in fields:
        if f == 'TRANSCRIPT':
            lines.append('TRANSCRIPT: <verbatim transcript of the audio, exact words only>')
        elif f == 'ROLE':
            lines.append("ROLE: <choose exactly ONE of: ATC, Pilot, Other -- HOW TO DECIDE: count who is speaking for MORE of the clip's total duration/word count -- the dominant speaker overall>")
        elif f == 'HIGH_INTENT':
            lines.append('HIGH_INTENT: <the single BROAD category, choose exactly one of these 8 words only: ' + ', '.join(HIGH_INTENT_CLASSES) + '. HOW TO DECIDE: a controller INSTRUCTING a pilot to do something new is Command or Clearance; a readback/repeat-back of an instruction is Confirmation, NOT Command; stating unprompted situational data is Report, sharing reference/advisory data is Information; asking for something is Request; imposing a limit is Restriction. Do NOT put a FINE_INTENT value here -- these are different fields.>')
        elif f == 'FINE_INTENT':
            lines.append('FINE_INTENT: <the single DETAILED category, choose exactly one from this fixed list only: ' + ', '.join(FINE_INTENT_CLASSES) + '. HOW TO DECIDE: pick the option consistent with your HIGH_INTENT answer and with what is literally said in THIS transmission. Do NOT put a HIGH_INTENT value here.>')
        elif f == 'SLOTS':
            if slot_columns:
                keys = sorted((c.replace('slot__', '') for c in slot_columns))
                lines.append('SLOTS: <checklist of possible fields -- only include a key if you can actually identify it from THIS audio clip. Most clips only have a few applicable keys. Output semicolon-separated key=value pairs using ONLY these exact key names: ' + ', '.join(keys) + ". WORKED EXAMPLE: 'callsign=Speedbird 123; altitude=5000 feet; heading=270'. Omit inapplicable keys entirely (do not write 'key=none'). Output the single word 'none' only if nothing on this checklist applies.>")
            else:
                lines.append("SLOTS: <semicolon-separated key=value pairs for any slots present, or 'none' if no slots apply>")
    return '\n'.join(lines)

def compute_slot_support(df, all_slot_columns):
    supported = []
    for col in all_slot_columns:
        if col in EXCLUDED_SLOT_COLUMNS:
            continue
        vals = df[col].astype(str).str.strip()
        vals = vals[~vals.str.lower().isin(['', 'nan', 'none'])]
        if len(vals) > 0:
            supported.append(col)
    return supported

def gold_roles_for_row(row):
    primary = clean(row[ROLE_COL])
    return {primary} if primary else set()
ROLE_LABEL_SYNONYMS = {'pilots': 'Pilot', 'pilot': 'Pilot', 'tower': 'ATC', 'atc': 'ATC', 'atco': 'ATC', 'other': 'Other'}

def parse_roles(role_text):
    role_text = clean(role_text)
    if not role_text:
        return set()
    out = set()
    for r in role_text.split(','):
        r = clean(r)
        if not r:
            continue
        out.add(ROLE_LABEL_SYNONYMS.get(r.lower(), r))
    return out

def set_metrics(gold_set, pred_set):
    gold_norm = {r.lower() for r in gold_set}
    pred_norm = {r.lower() for r in pred_set}
    matched = len(gold_norm & pred_norm)
    precision = matched / len(pred_norm) if pred_norm else 1.0 if not gold_norm else 0.0
    recall = matched / len(gold_norm) if gold_norm else 1.0 if not pred_norm else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    exact = gold_norm == pred_norm
    return (matched, precision, recall, f1, exact)

def gold_slots_for_row(row, slot_columns):
    out = {}
    for col in slot_columns:
        val = row.get(col, None)
        if pd.isna(val):
            continue
        val = str(val).strip()
        if not val or val.lower() in ('nan', 'none'):
            continue
        key = col.replace('slot__', '')
        out[key] = clean_transcript(val)
    return out

def build_target(row, stage, slot_columns):
    lines = [f"TRANSCRIPT: {clean_transcript(row['full_transcript'])}"]
    if stage in ('role', 'role_intent', 'role_intent_slot'):
        roles = gold_roles_for_row(row)
        lines.append(f"ROLE: {(', '.join(sorted(roles)) if roles else clean(row[ROLE_COL]))}")
    if stage in ('role_intent', 'role_intent_slot'):
        lines.append(f'HIGH_INTENT: {clean(row[HIGH_INTENT_COL])}')
        lines.append(f'FINE_INTENT: {clean(row[FINE_INTENT_COL])}')
    if stage == 'role_intent_slot':
        slots = gold_slots_for_row(row, slot_columns)
        if slots:
            slot_text = '; '.join((f'{k}={v}' for (k, v) in sorted(slots.items())))
        else:
            slot_text = 'none'
        lines.append(f'SLOTS: {slot_text}')
    return '\n'.join(lines)

def parse_structured_output(text, stage):
    fields = stage_fields(stage)
    positions = []
    for f in fields:
        idx = text.find(f + ':')
        if idx != -1:
            positions.append((idx, f))
    positions.sort()
    result = {f: '' for f in fields}
    for (i, (idx, f)) in enumerate(positions):
        start = idx + len(f) + 1
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        result[f] = text[start:end].strip()
    return result

def parse_slots(slots_text):
    slots_text = clean(slots_text)
    if not slots_text or slots_text.lower() == 'none':
        return {}
    out = {}
    for part in slots_text.split(';'):
        part = part.strip()
        if not part or '=' not in part:
            continue
        (k, v) = part.split('=', 1)
        out[clean(k)] = clean(v)
    return out

def slot_3part_counts(gold_slots, pred_slots, all_keys):
    gold_keys = set(gold_slots.keys())
    pred_keys = set(pred_slots.keys())
    non_gold_keys = set(all_keys) - gold_keys
    identify_hit = len(gold_keys & pred_keys)
    identify_total = len(gold_keys)
    omit_hit = len(non_gold_keys - pred_keys)
    omit_total = len(non_gold_keys)
    value_hit = sum((1 for k in gold_keys & pred_keys if pred_slots[k].strip().lower() == gold_slots[k].strip().lower()))
    value_total = len(gold_keys & pred_keys)
    return (identify_hit, identify_total, omit_hit, omit_total, value_hit, value_total)

def make_splits(metadata_csv, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    df = pd.read_csv(metadata_csv, low_memory=False)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return (df, train_df, val_df, test_df)

def choose_subset(df, n, seed):
    if n <= 0 or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed)

def build_conversation(audio_path, prompt_text):
    return [{'role': 'user', 'content': [{'type': 'audio', 'path': audio_path}, {'type': 'text', 'text': prompt_text}]}]

def build_vanilla_model(device):
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(VOXTRAL_MODEL, local_files_only=True)
    if PRECISION == 'bf16':
        model = VoxtralForConditionalGeneration.from_pretrained(VOXTRAL_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='balanced', local_files_only=True)
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = VoxtralForConditionalGeneration.from_pretrained(VOXTRAL_MODEL, quantization_config=bnb_config, device_map={'': device}, local_files_only=True)
        model.gradient_checkpointing_enable()
    real_device = next(model.parameters()).device
    return (model, processor, real_device)

def apply_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if PRECISION != 'bf16':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, task_type='CAUSAL_LM')
    return get_peft_model(model, config)

def load_lora_checkpoint(base_device, adapter_dir, is_trainable=True):
    from peft import PeftModel, prepare_model_for_kbit_training
    (base_model, processor, real_device) = build_vanilla_model(base_device)
    if is_trainable and PRECISION != 'bf16':
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=is_trainable)
    return (model, processor, real_device)

def build_or_resume(device, output_dir, warm_start_ckpt=None, resume=False):
    output_dir = Path(output_dir)
    latest = output_dir / 'LATEST_MIDEPOCH'
    best = output_dir / 'GLOBAL_BEST'
    if resume and latest.exists() and any(latest.iterdir()):
        print(f'--resume set: resuming from mid-epoch checkpoint: {latest}')
        return load_lora_checkpoint(device, latest, is_trainable=True)
    if resume and best.exists() and any(best.iterdir()):
        print(f'--resume set: resuming from best checkpoint: {best}')
        return load_lora_checkpoint(device, best, is_trainable=True)
    if not resume and (latest.exists() and any(latest.iterdir()) or (best.exists() and any(best.iterdir()))):
        print(f'Existing checkpoint found at {output_dir} but --resume not set: starting fresh.')
    if warm_start_ckpt is not None:
        print(f'Warm-starting from previous-stage checkpoint: {warm_start_ckpt}')
        return load_lora_checkpoint(device, warm_start_ckpt, is_trainable=True)
    (model, processor, device) = build_vanilla_model(device)
    model = apply_lora(model)
    return (model, processor, device)

@torch.no_grad()
def run_inference(model, processor, audio_path, prompt_text, device, max_new_tokens=MAX_NEW_TOKENS_DEFAULT):
    conv = build_conversation(audio_path, prompt_text)
    inputs = processor.apply_chat_template(conv, add_generation_prompt=True, return_tensors='pt', return_dict=True)
    inputs = inputs.to(device=device)
    prompt_len = inputs['input_ids'].shape[1]
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.3)
    text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0]
    return text
SEMANTIC_JUDGE_PROMPT = 'You are an aviation ATC communications expert reviewing a disagreement\nbetween two possible intent labels for a radio transcript.\n\nTRANSCRIPT: {transcript}\n\nLABEL A: {label_a}\nLABEL B: {label_b}\n\nQuestion: for THIS specific transcript, are Label A and Label B effectively describing the\nSAME communicative intent (just different wording of the same idea), a RELATED but distinct\nintent (a plausible alternate reading of the same utterance, partial overlap), or a genuinely\nDIFFERENT intent (they describe different things being communicated)?\n\nAnswer with exactly one word: SAME, RELATED, or DIFFERENT.'

@torch.no_grad()
def judge_semantic_relation(model, processor, device, transcript, label_a, label_b):
    prompt = SEMANTIC_JUDGE_PROMPT.format(transcript=transcript, label_a=label_a, label_b=label_b)
    conv = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(conv, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device=device)
    plen = inputs['input_ids'].shape[1]
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    text = processor.batch_decode(out[:, plen:], skip_special_tokens=True)[0].strip().upper()
    if 'SAME' in text:
        return (1.0, text)
    if 'RELATED' in text:
        return (0.5, text)
    return (0.0, text)

def evaluate_df(model, processor, df, device, stage, prompt_text, slot_columns, max_samples=None, seed=0, log_prefix='', max_new_tokens=MAX_NEW_TOKENS_DEFAULT):
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed)
    model.eval()
    rows = []
    all_slot_keys = sorted((c.replace('slot__', '') for c in slot_columns)) if slot_columns else []
    id_hit_t = id_tot_t = om_hit_t = om_tot_t = val_hit_t = val_tot_t = 0
    pbar = tqdm(df.iterrows(), total=len(df), desc=log_prefix or 'eval', mininterval=5.0, dynamic_ncols=True)
    for (_, row) in pbar:
        gold_transcript = clean_transcript(row['full_transcript'])
        t0 = time.time()
        raw_out = run_inference(model, processor, row['audio_path'], prompt_text, device, max_new_tokens)
        elapsed = time.time() - t0
        parsed = parse_structured_output(raw_out, stage)
        pred_transcript = clean(parsed.get('TRANSCRIPT', ''))
        duration = row.get('audio_duration_sec', None)
        rtf = elapsed / duration if duration and duration > 0 else None
        wer = word_error_rate(gold_transcript, pred_transcript)
        record = {'audio_name': row['audio_name'], 'transcript_gold': gold_transcript, 'transcript_pred': pred_transcript, 'WER': wer, 'CER': char_error_rate(gold_transcript, pred_transcript), 'sacreBLEU': sacrebleu.sentence_bleu(pred_transcript, [gold_transcript]).score if HAVE_BLEU else None, 'Cosine': COSINE_FN(gold_transcript, pred_transcript), 'ROUGE_L': rouge_l(gold_transcript, pred_transcript), 'TokenF1': token_f1(gold_transcript, pred_transcript), 'RTF': rtf}
        if stage in ('role', 'role_intent', 'role_intent_slot'):
            gold_roles = gold_roles_for_row(row)
            pred_roles = parse_roles(parsed.get('ROLE', ''))
            (r_matched, r_precision, r_recall, r_f1, r_exact) = set_metrics(gold_roles, pred_roles)
            record['role_gold'] = ', '.join(sorted(gold_roles)) if gold_roles else 'none'
            record['role_pred'] = ', '.join(sorted(pred_roles)) if pred_roles else 'none'
            record['role_match'] = r_exact
        if stage in ('role_intent', 'role_intent_slot'):
            hi_gold = clean(row[HIGH_INTENT_COL])
            hi_pred = clean(parsed.get('HIGH_INTENT', ''))
            fi_gold = clean(row[FINE_INTENT_COL])
            fi_pred = clean(parsed.get('FINE_INTENT', ''))
            record['high_intent_gold'] = hi_gold
            record['high_intent_pred'] = hi_pred
            record['fine_intent_gold'] = fi_gold
            record['fine_intent_pred'] = fi_pred
            exact_hi = hi_gold.lower() == hi_pred.lower()
            if exact_hi or not hi_pred:
                hi_score = 1.0 if exact_hi else 0.0
            else:
                (hi_score, _) = judge_semantic_relation(model, processor, device, gold_transcript, hi_gold, hi_pred)
            exact_fi = fi_gold.lower() == fi_pred.lower()
            if exact_fi or not fi_pred:
                fi_score = 1.0 if exact_fi else 0.0
            else:
                (fi_score, _) = judge_semantic_relation(model, processor, device, gold_transcript, fi_gold, fi_pred)
            record['High Intent Accuracy'] = hi_score
            record['Fine Intent Accuracy'] = fi_score
        if stage == 'role_intent_slot':
            gold_slots = gold_slots_for_row(row, slot_columns)
            pred_slots = parse_slots(parsed.get('SLOTS', ''))
            record['slot_gold'] = json.dumps(gold_slots, ensure_ascii=False)
            record['slot_pred'] = json.dumps(pred_slots, ensure_ascii=False)
            (ih, it, oh, ot, vh, vt) = slot_3part_counts(gold_slots, pred_slots, all_slot_keys)
            id_hit_t += ih
            id_tot_t += it
            om_hit_t += oh
            om_tot_t += ot
            val_hit_t += vh
            val_tot_t += vt
        rows.append(record)
        pbar.set_postfix(wer=f'{wer:.3f}')
    pbar.close()
    d = pd.DataFrame(rows)
    metrics = {'WER': d['WER'].mean(), 'CER': d['CER'].mean(), 'sacreBLEU': d['sacreBLEU'].mean() if HAVE_BLEU else None, 'Cosine': d['Cosine'].mean(), 'ROUGE_L': d['ROUGE_L'].mean(), 'TokenF1': d['TokenF1'].mean(), 'RTF': d['RTF'].mean() if d['RTF'].notna().any() else None}
    if 'role_match' in d.columns:
        metrics['role_acc'] = d['role_match'].mean()
    if 'High Intent Accuracy' in d.columns:
        metrics['High Intent Accuracy'] = d['High Intent Accuracy'].mean()
        metrics['Fine Intent Accuracy'] = d['Fine Intent Accuracy'].mean()
    if stage == 'role_intent_slot':
        identify_acc = id_hit_t / id_tot_t if id_tot_t else 1.0
        omit_acc = om_hit_t / om_tot_t if om_tot_t else 1.0
        value_acc = val_hit_t / val_tot_t if val_tot_t else 1.0
        metrics['Slot Accuracy'] = (identify_acc + omit_acc + value_acc) / 3.0
    return (metrics, d)

def compute_class_weights(train_df, col):
    counts = train_df[col].astype(str).str.strip().replace('', pd.NA).dropna().value_counts()
    if counts.empty:
        return {}
    inv = 1.0 / counts
    weighted_mean = (inv * counts).sum() / counts.sum()
    weights = (inv / weighted_mean).to_dict()
    return {str(k): float(v) for (k, v) in weights.items()}

def compute_slot_key_weights(train_df, slot_columns):
    weights = {}
    supports = {}
    for col in slot_columns:
        key = col.replace('slot__', '')
        vals = train_df[col].astype(str).str.strip()
        support = (~vals.str.lower().isin(['', 'nan', 'none'])).sum()
        supports[key] = support
    if not supports:
        return weights
    total_support = sum(supports.values())
    for (key, support) in supports.items():
        weights[key] = 1.0 if support == 0 else 1.0 / support
    weighted_mean = sum((weights[k] * supports[k] for k in weights)) / max(1, total_support)
    if weighted_mean > 0:
        weights = {k: v / weighted_mean for (k, v) in weights.items()}
    return weights

class OversampleWeights:

    def __init__(self, train_df, stage, slot_columns):
        self.role_w = compute_class_weights(train_df, ROLE_COL) if stage in ('role', 'role_intent', 'role_intent_slot') else {}
        self.high_w = compute_class_weights(train_df, HIGH_INTENT_COL) if stage in ('role_intent', 'role_intent_slot') else {}
        self.fine_w = compute_class_weights(train_df, FINE_INTENT_COL) if stage in ('role_intent', 'role_intent_slot') else {}
        self.slot_w = compute_slot_key_weights(train_df, slot_columns) if stage == 'role_intent_slot' else {}

    def row_weight(self, row, slot_columns):
        w = 1.0
        if self.role_w:
            w *= self.role_w.get(clean(row[ROLE_COL]), 1.0)
        if self.high_w:
            w *= self.high_w.get(clean(row[HIGH_INTENT_COL]), 1.0)
        if self.fine_w:
            w *= self.fine_w.get(clean(row[FINE_INTENT_COL]), 1.0)
        if self.slot_w:
            gold_slots = gold_slots_for_row(row, slot_columns)
            if gold_slots:
                key_ws = [self.slot_w.get(k, 1.0) for k in gold_slots]
                w *= sum(key_ws) / len(key_ws)
        return max(0.5, min(3.0, w))

def build_oversampled_order(train_df, weight_state, slot_columns, max_repeat=OVERSAMPLE_MAX_REPEAT):
    idx_list = []
    for (idx, row) in train_df.iterrows():
        w = weight_state.row_weight(row, slot_columns)
        repeat = max(1, min(max_repeat, int(round(w))))
        idx_list.extend([idx] * repeat)
    random.shuffle(idx_list)
    return idx_list

def compute_train_loss(model, processor, row, prompt_text, target_text, device):
    conv = build_conversation(row['audio_path'], prompt_text)
    prompt_inputs = processor.apply_chat_template(conv, add_generation_prompt=True, return_tensors='pt', return_dict=True)
    prompt_inputs = prompt_inputs.to(device=device)
    MAX_TARGET_TOKENS = 1024
    target_ids = processor.tokenizer(target_text, return_tensors='pt', add_special_tokens=False, truncation=True, max_length=MAX_TARGET_TOKENS).input_ids.to(device)
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = torch.cat([target_ids, torch.tensor([[eos_id]], device=device)], dim=1)
    prompt_ids = prompt_inputs['input_ids']
    prompt_mask = prompt_inputs['attention_mask']
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, torch.ones_like(target_ids)], dim=1)
    labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)
    extra_kwargs = {k: v for (k, v) in prompt_inputs.items() if k not in ('input_ids', 'attention_mask')}
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **extra_kwargs)
    return out.loss

def train_one_epoch(model, processor, optimizer, train_df, stage, prompt_text, slot_columns, device, weight_state, grad_accum=4, desc='train', mid_ckpt_dir=None, checkpoint_every=200):
    model.train()
    order = build_oversampled_order(train_df, weight_state, slot_columns)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    (losses, skipped) = ([], 0)
    pbar = tqdm(enumerate(order, 1), total=len(order), desc=desc, mininterval=5.0, dynamic_ncols=True)
    for (step, idx) in pbar:
        row = train_df.loc[idx]
        target_text = build_target(row, stage, slot_columns)
        loss = compute_train_loss(model, processor, row, prompt_text, target_text, device)
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss='skip', skipped=skipped)
            continue
        (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == len(order):
            if all((torch.isfinite(p.grad).all() for p in trainable_params if p.grad is not None)):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        pbar.set_postfix(loss=f'{losses[-1]:.3f}', avg=f'{sum(losses) / len(losses):.3f}', skipped=skipped)
        if mid_ckpt_dir and checkpoint_every and (step % checkpoint_every == 0):
            Path(mid_ckpt_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(mid_ckpt_dir))
    pbar.close()
    return (sum(losses) / max(1, len(losses)), skipped)

def run_stage(model, processor, train_df, val_df, stage, prompt_text, slot_columns, epochs, patience, lr, output_dir, train_eval_subsample, stage_label, device, grad_accum, weight_source_df=None):
    print(f'[{stage_label}] train={len(train_df)} validation={len(val_df)}  methodology=oversample')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'PROMPT_USED.txt', 'w') as f:
        f.write(prompt_text)
    weight_state = OversampleWeights(weight_source_df if weight_source_df is not None else train_df, stage, slot_columns)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    (best_wer, epochs_since_improvement, history) = (float('inf'), 0, [])
    mid_ckpt_dir = output_dir / 'LATEST_MIDEPOCH'
    for epoch in range(1, epochs + 1):
        (train_loss, skipped) = train_one_epoch(model, processor, optimizer, train_df, stage, prompt_text, slot_columns, device, weight_state=weight_state, grad_accum=grad_accum, desc=f'{stage_label} ep{epoch} train', mid_ckpt_dir=mid_ckpt_dir, checkpoint_every=200)
        (train_metrics, _) = evaluate_df(model, processor, train_df, device, stage, prompt_text, slot_columns, max_samples=train_eval_subsample, seed=epoch, log_prefix=f'{stage_label} ep{epoch} train-eval')
        (val_metrics, val_d) = evaluate_df(model, processor, val_df, device, stage, prompt_text, slot_columns, log_prefix=f'{stage_label} ep{epoch} val-eval')
        record = {'stage': stage_label, 'epoch': epoch, 'train_loss': train_loss, 'skipped': skipped}
        for (k, v) in train_metrics.items():
            record[f'train_{k}'] = v
        for (k, v) in val_metrics.items():
            record[f'val_{k}'] = v
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / 'HISTORY.csv', index=False)
        print(record)
        improved = val_metrics['WER'] < best_wer
        if improved:
            (best_wer, epochs_since_improvement) = (val_metrics['WER'], 0)
            best_dir = output_dir / 'GLOBAL_BEST'
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
        else:
            epochs_since_improvement += 1
        if patience > 0 and epochs_since_improvement >= patience:
            print(f'[{stage_label}] early stopping at epoch {epoch}, best_wer={best_wer:.4f}')
            break
    return (output_dir / 'GLOBAL_BEST', best_wer)

def test_and_export(model, processor, test_df, stage, prompt_text, slot_columns, output_dir, stage_label, device, test_n=0):
    if test_n and test_n < len(test_df):
        test_df = test_df.sample(n=test_n, random_state=99)
    (metrics, test_pred) = evaluate_df(model, processor, test_df, device, stage, prompt_text, slot_columns, log_prefix=f'{stage_label} test')
    print(f'[{stage_label}] TEST metrics: {metrics}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_pred['Pipeline'] = stage_label
    test_pred.to_csv(output_dir / f'full_test_{stage_label}.csv', index=False)
    summary_row = {'stage': stage_label, 'n_test_samples': len(test_pred)}
    summary_row.update(metrics)
    pd.DataFrame([summary_row]).to_csv(output_dir / 'test_summary.csv', index=False)
    return metrics
ORIGINAL_ASR_PROMPT = 'You are an aviation expert now transcribe this audio only provide the text transcribe nothing more than that.'

@torch.no_grad()
def check_asr_retention(model, processor, test_df, device, n=15, seed=123):
    model.eval()
    sample_df = test_df.sample(n=min(n, len(test_df)), random_state=seed)
    wers = []
    for (_, row) in sample_df.iterrows():
        gold = clean_transcript(row['full_transcript'])
        conv = build_conversation(row['audio_path'], ORIGINAL_ASR_PROMPT)
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device=device)
        prompt_len = inputs['input_ids'].shape[1]
        out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, no_repeat_ngram_size=3, repetition_penalty=1.3)
        pred = clean(processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0])
        wers.append(word_error_rate(gold, pred))
    return sum(wers) / len(wers) if wers else None

def main():
    global PRECISION
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'full'], default='full')
    ap.add_argument('--precision', choices=['4bit', 'bf16'], default='4bit')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--metadata', default=METADATA_CSV)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    args = ap.parse_args()
    PRECISION = args.precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (full_df, train_df, val_df, test_df) = make_splits(args.metadata)
    all_slot_columns = [c for c in full_df.columns if c.startswith('slot__')]
    slot_columns = compute_slot_support(full_df, all_slot_columns) if STAGE == 'role_intent_slot' else []
    print(f'Split sizes -> train={len(train_df)} val={len(val_df)} test={len(test_df)} (ratio {TRAIN_FRAC}/{VAL_FRAC}/{TEST_FRAC}, seed={SPLIT_SEED})')
    if args.mode == 'smoke':
        hp = {'train_n': 6, 'val_n': 4, 'epochs': 2, 'test_n': 6, 'train_eval_subsample': 4, 'lr': 5e-05, 'grad_accum': 1}
    else:
        hp = {'train_n': 0, 'val_n': 0, 'epochs': 10, 'test_n': 0, 'train_eval_subsample': 100, 'lr': 5e-05, 'grad_accum': 4}
    full_train_df_for_weights = train_df
    train_df = choose_subset(train_df, hp['train_n'], 42)
    val_df = choose_subset(val_df, hp['val_n'], 43)
    print('=' * 70)
    print(f'MODE: {args.mode}  STAGE: {STAGE}  STRATEGY: {STRATEGY}  DEVICE: {device}  METHODOLOGY: oversample')
    print('=' * 70)
    prompt_text = build_prompt(STAGE, slot_columns)
    out_dir = Path(args.output_dir)
    warm_start_ckpt = ROLE_INTENT_CHECKPOINT
    if not Path(warm_start_ckpt).exists():
        raise SystemExit(f'Run ATCO2_roletointent_sequential_ov.py first -- checkpoint not found at {warm_start_ckpt}')
    print(f'Building Voxtral base model + LoRA ({STAGE_LABEL}), resume={args.resume}...')
    (model, processor, device) = build_or_resume(device, out_dir, warm_start_ckpt=warm_start_ckpt, resume=args.resume)
    (best_ckpt, best_wer) = run_stage(model, processor, train_df, val_df, STAGE, prompt_text, slot_columns, hp['epochs'], patience=3, lr=hp['lr'], output_dir=out_dir, train_eval_subsample=hp['train_eval_subsample'], stage_label=STAGE_LABEL, device=device, grad_accum=hp['grad_accum'], weight_source_df=full_train_df_for_weights)
    del model
    torch.cuda.empty_cache()
    (model, processor, device) = load_lora_checkpoint(device, best_ckpt, is_trainable=False)
    test_and_export(model, processor, test_df, STAGE, prompt_text, slot_columns, out_dir, STAGE_LABEL, device, test_n=hp['test_n'])
    print('Checking ASR retention (plain transcribe-only prompt, to catch forgetting)...')
    retention_wer = check_asr_retention(model, processor, test_df, device)
    print(f'[{STAGE_LABEL}] ASR-RETENTION-CHECK WER (plain prompt): {retention_wer:.4f} -- compare to the original ASR-only baseline (~0.25 test WER).')
    print(f'DONE {STAGE_LABEL}. best_val_wer:', best_wer)
if __name__ == '__main__':
    main()
