"""CPU-only: source patch provenance and no DEV-label leakage before frozen selection."""
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
src=root/'source/01_risk_adaptive_dynamic_k.py'
s=src.read_text()
assert s.count('PT108_ADAPTER_BANK=None  # original path when disabled\n')==1
assert s.count('PT108_ADAPTER_BANK[self.router_key](xf)')==1
assert s.count('self.last_pt108_kl_loss=F.kl_div(')==1
s=s.replace('PT108_ADAPTER_BANK=None  # original path when disabled\n','',1)
s=s.replace('''            original_logits=router(xf)
            logits=original_logits
            if PT108_ADAPTER_BANK is not None:
                logits=original_logits + PT108_ADAPTER_BANK[self.router_key](xf).to(original_logits.dtype)
            clogit=self.crit_head()(xf).squeeze(-1)''','''            logits=router(xf)
            clogit=self.crit_head()(xf).squeeze(-1)''',1)
s=s.replace('''            if PT108_ADAPTER_BANK is not None:
                reference_probability=torch.softmax(original_logits.detach().float(),dim=-1)
                self.last_pt108_kl_loss=F.kl_div(torch.log_softmax(logits.float(),dim=-1),reference_probability,reduction='batchmean')
            else:
                self.last_pt108_kl_loss=None
''','',1)
original=hashlib.sha256(s.encode()).hexdigest()
assert original=='9a726c82fc753c11ab49407b8a11ccd2e3a7af777b3ee164d546b8762a18e881',original
runner=(root/'pt108_matched_router_train.py').read_text()
assert runner.count('pd.read_csv(DEV,low_memory=False)')==1
assert runner.index("write_json(OUT/'FROZEN_BEFORE_DEV61.json',frozen)")<runner.index('pd.read_csv(DEV,low_memory=False)')
assert "if not valid:" in runner and "'dev61_candidate_evaluated':False" in runner
assert "EPOCHS=1" in runner and "MODES=('omni','hierarchical')" in runner
print('PT108_SOURCE_TEST_PASS: original source SHA parity, minimal adapter patch, DEV61 read after immutable freeze only')
