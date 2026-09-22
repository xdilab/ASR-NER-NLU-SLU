# PT108 routing adapters — source review

**Scope:** frozen-backbone, frozen-expert ASR experiment comparing two *residual logit parameterizations*. This package is a source-review snapshot of the PT108 code plus two earlier inference-only routing prototypes. It is not a standalone dataset or checkpoint distribution.

## Read these files first

| File | Purpose |
|---|---|
| `pt108_adapters.py` | Flat residual adapter (`OmniResidual`), grouped-logit adapter (`HierarchicalResidual`), deterministic per-router pair selection. |
| `pt108_matched_router_train.py` | Training loop, initialization/parity, weight isolation, calibration gates, frozen selection, paired evaluation. |
| `source/01_risk_adaptive_dynamic_k.py` | Existing PT57 dynamic-k routing module with PT108 residual-logit and KL hooks. |
| `pt61_reference.py` and `source/02_pt57g_asr_specialized_train.py` | Model-loading, original checkpoint and evaluation dependencies. |
| `run_pt108_gpu0.sh` | Workstation launcher. Requires the *full* PT108 experiment package, datasets, checkpoint, and local Python environment. |
| `test_pt108_cpu.py` / `test_pt108_source.py` | Synthetic gradient/capacity checks and exact source-patch provenance/DEV-read-order checks. |
| `prior_prototypes/` | PT104 fixed two-group and PT106 selective-affinity inference-only policies, shown separately from PT108. |

## What the two PT108 adapters actually compute

Both retain the original PT57 expert weights, trained router weights, dynamic-k budget and global top-k dispatcher. At router group `j`, original logits `z_j(x)` receive an additive learned residual.

**Flat / Omni-style prototype:** `z'_j(x) = z_j(x) + W_j x`, with `W_j` a 4-by-d matrix, zero-initialized. This is a local flat routing-logit update; it is *not* a complete Omni Router implementation.

**Grouped-logit / hierarchical prototype:** each local router has a deterministic pairing of four experts into two pairs, selected from the original router-weight cosine similarities. Two linear heads produce group scores `g0, g1` and within-pair contrasts `f0, f1`; residuals in paired order are `(g0+f0, g0-f0, g1+f1, g1-f1)` and are restored to the original expert order before PT57's unchanged global top-k dispatch. This is **not** conditional group-then-expert dispatch or a fully trained hierarchical MoE.

**Important interpretation limit:** the above four-dimensional linear transform is invertible. Consequently, the grouped and flat adapters have the *same linear residual-logit function class* and the same parameter count (4d per local router). Differences, if observed, would be attributable to parameterization, optimization, pairing and training dynamics—not an expanded hierarchical representational capacity. A true conditional hierarchy would require a separate implementation and matched compute evaluation.

## Experimental controls and status

- Both adapters initialize at zero: before training they reproduce PT57's original routing logits.
- TRAIN225 supplies optimization examples; TRAIN56 supplies the gate. PT57 had originally been trained on all TRAIN281, so TRAIN56 is **not** independent of the base model. DEV61 is read only after the winner is frozen, and has been inspected in earlier experiments. The historical 175 is not used by this script.
- Accept a candidate for exploratory DEV61 comparison only if TRAIN56 WER strictly improves, critical-word accuracy is maintained, the budget certificate passes, and average active experts stay within 0.04 of the baseline.
- No PT108 GPU outcome is included here. Prior PT104/PT106 grouping experiments did not displace PT57. Do not report a PT108 gain without its completed results.

## Source verification

The executable PT108 source and the included prior-prototype Python files are copied byte-for-byte from their experiment archives. See `SOURCE_MANIFEST.json` for SHA-256 checksums. No model, audio, prediction, reference or label files are included. CPU tests can be run in an environment with PyTorch using `python test_pt108_cpu.py` and `python test_pt108_source.py` from this folder. To reproduce the *full experiment*, use the full PT108 package, its original inputs, the frozen PT57 checkpoint and the workstation environment described in the original run README.

The code contains experimental assumptions and hardcoded workstation paths; review those before adapting it to another system. This snapshot does not assert authorship or a development history.
