"""PT108 small matched trainable residual routing heads; NO gold input at inference.
Both modes contain exactly four input-dimension weights per PT57 router group.
The hierarchical variant factorizes those four into a 2-way coarse group head
and two 1-logit within-group decisions. Dispatch is still PT57 top-k: this is
NOT a fully conditional hierarchical MoE and is described as a prototype.
"""
import torch
from torch import nn
from torch.nn import functional as F

PAIRS = (((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))

class OmniResidual(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.flat=nn.Linear(dim,4,bias=False,device=device,dtype=torch.float32)
        nn.init.zeros_(self.flat.weight)
    def forward(self,x):
        return self.flat(x)

class HierarchicalResidual(nn.Module):
    def __init__(self, dim, device, pairing):
        super().__init__()
        self.coarse=nn.Linear(dim,2,bias=False,device=device,dtype=torch.float32)
        self.within=nn.Linear(dim,2,bias=False,device=device,dtype=torch.float32)
        nn.init.zeros_(self.coarse.weight)
        nn.init.zeros_(self.within.weight)
        order=list(pairing[0]+pairing[1])
        assert sorted(order)==[0,1,2,3]
        self.register_buffer('inverse',torch.tensor([order.index(j) for j in range(4)],device=device,dtype=torch.long),persistent=False)
    def forward(self,x):
        g=self.coarse(x); f=self.within(x)
        original_order=torch.stack((g[:,0]+f[:,0],g[:,0]-f[:,0],g[:,1]+f[:,1],g[:,1]-f[:,1]),dim=-1)
        return original_order.index_select(-1,self.inverse)

def weight_only_pairing(original_router):
    # Deterministic, no audio / labels / development metrics; compares PT57 original weights.
    w=original_router.weight.detach().float()
    w=F.normalize(w,dim=-1,eps=1e-12)
    sim=w @ w.T
    scores=[float(sim[a,b]+sim[c,d]) for (a,b),(c,d) in PAIRS]
    idx=max(range(3),key=lambda i:(scores[i],-i))
    return PAIRS[idx], scores

def make_bank(model, mode):
    assert mode in ('omni','hierarchical')
    routerbank=model.moe_router_bank
    modules=nn.ModuleDict(); pairing_meta={}
    for key,router in routerbank.items():
        dim=int(router.in_features); dev=router.weight.device
        if mode=='omni':
            modules[key]=OmniResidual(dim,dev)
        else:
            pairing,scores=weight_only_pairing(router)
            modules[key]=HierarchicalResidual(dim,dev,pairing)
            pairing_meta[key]={'pairing':[list(pairing[0]),list(pairing[1])], 'router_weight_cosine_scores':scores}
    expected=sum(int(r.in_features)*4 for r in routerbank.values())
    count=sum(p.numel() for p in modules.parameters())
    if len(modules)!=112 or count!=expected:
        raise RuntimeError(f'Matched capacity gate failed: {len(modules)} router groups, {count} params expected {expected}')
    if any(p.detach().abs().max().item()!=0 for p in modules.parameters()):
        raise RuntimeError('Residual adapters must start from exact zero')
    return modules,pairing_meta,count
