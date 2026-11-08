import os,json,time
from pathlib import Path
import torch
from common_v2 import Trajectories
from model import make_model
from dynamics import loss_terms
p=Path(os.environ['V2_ROOT']);out=p/'reports/slow-gradient-probe';out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(16);begun=time.time();rows=[]
for step in [25000,65000]:
 src=p/'snapshots/hierarchical-seed43'/f'checkpoint-step-{step:07d}.pt'
 state=torch.load(src,map_location='cpu',weights_only=False);c=state['config']
 model=make_model('hierarchical',c['width'],c['slow_layers'],c['fast_layers'],c['heads'])
 model.load_state_dict(state['model']);model.load_state_dict(state.get('ema') or state['model'],strict=False);del state
 model.eval()
 for q in model.fast.parameters():q.requires_grad_(False)
 params=list(model.slow.parameters())
 for split in ['train','val']:
  data=Trajectories(split,'cpu')
  for i in range(4):
   batch,meta=data.batch(1,257,81000+i,balanced=True,name_offset=i);torch.manual_seed(71000+i)
   _,n,_,_=loss_terms(model,batch,step,True)
   geom=n['bond']+.1*n['clash']+.01*n['torsion']
   a=torch.autograd.grad(n['slow_fm'],params,retain_graph=True,allow_unused=True)
   b=torch.autograd.grad(geom,params,allow_unused=True)
   aa=sum(float(v.square().sum()) for v in a if v is not None);bb=sum(float(v.square().sum()) for v in b if v is not None)
   dot=sum(float((u*v).sum()) for u,v in zip(a,b) if u is not None and v is not None)
   row=dict(step=step,split=split,sample=meta,losses={k:float(v.detach()) for k,v in n.items()},slow_fm_gradient_norm=aa**.5,geometry_gradient_norm=bb**.5,gradient_cosine=dot/max((aa*bb)**.5,1e-20),elapsed_seconds=time.time()-begun)
   rows.append(row);(out/'results.json').write_text(json.dumps(dict(status='running',device='cpu',precision='fp32',scope='4 fixed clips per split; mechanistic gradient probe, not quality evaluation',rows=rows),indent=2));print(json.dumps(row),flush=True)
   del a,b,n,geom
  del data
 del model
(out/'results.json').write_text(json.dumps(dict(status='complete',device='cpu',precision='fp32',scope='4 fixed clips per split; CPU FP32 differs from training BF16; no causal attribution from cosine alone',rows=rows),indent=2))
