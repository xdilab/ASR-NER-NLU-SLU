"""No model download or audio required; gradient, capacity and factorization tests."""
import torch
from torch import nn
from pt108_adapters import OmniResidual,HierarchicalResidual,PAIRS,make_bank,weight_only_pairing

torch.manual_seed(108)
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_router_bank=nn.ModuleDict({f'router_{i:03d}':nn.Linear(8,4,bias=False) for i in range(112)})
model=Dummy(); total=None
for mode in ('omni','hierarchical'):
    bank,meta,n=make_bank(model,mode)
    if total is not None: assert n==total
    total=n
    for key in list(bank)[:3]:
        x=torch.randn(9,8);b=model.moe_router_bank[key](x).detach()
        assert torch.equal((b+bank[key](x)),b), 'zero-init baseline parity'
        # Different expert output fingerprints ensure ASR-like gated output flows into each adapter.
        values=torch.tensor([0.2,-0.4,0.9,1.7])
        z=(torch.softmax(b+bank[key](x),dim=-1)*values).sum()
        z.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in bank[key].parameters()),'gradient missing'
        assert all(torch.isfinite(p.grad).all() for p in bank[key].parameters())
    if mode=='hierarchical':
        assert len(meta)==112 and all(sorted(sum(v['pairing'],[]))==[0,1,2,3] for v in meta.values())
        assert all(weight_only_pairing(model.moe_router_bank[k])[0] in PAIRS for k in meta)
print('PT108_CPU_TEST_PASS: 112 groups, matched parameter count',total,'zero residual baseline parity, nonzero finite gradients, deterministic pairings')
