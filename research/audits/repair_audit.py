import json,sys,os
from pathlib import Path
import numpy as np
import torch
from common_v2 import Trajectories
from model import make_model
from evaluation import free_validation,metrics
# common_v2 prepends the upstream MDGen repo; import OUR trainer by explicit path.
import importlib.util
_spec=importlib.util.spec_from_file_location("rsi_audit_trainer",Path(os.environ['V2_ROOT'])/'runtime'/'train.py')
_trainer=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_trainer)
validate=_trainer.validate
assert validate.__module__ == "rsi_audit_trainer"
from figures import validation_figures,trajectory_video
p=Path(os.environ['V2_ROOT']);arm=sys.argv[1];assert arm in ['slow_dropout','geometry_detach_slow']
out=p/'reports'/f'repair-{arm}-65k';out.mkdir(parents=True,exist_ok=True)
src=p/'runs'/f'hierarchical-seed43-{arm}-65k/checkpoint-step-0065000.pt'
s=torch.load(src,map_location='cpu',weights_only=False);assert s['step']==65000;c=s['config'];assert c['repair']==arm
m=make_model('hierarchical',c['width'],c['slow_layers'],c['fast_layers'],c['heads']).cuda();m.load_state_dict(s['model']);m.load_state_dict(s.get('ema') or s['model'],strict=False);del s
m.eval();data=Trajectories('val','cuda');loss,_=validate(m,data,256,65000)
pooled,a=free_validation(m,data,256,out,65000,32,16,32);validation_figures(out,a);trajectory_video(out,a)
meta=json.loads((out/'metrics.json').read_text())['sample_info'];names=[n for row in meta for n in row['names']]
def per_molecule(arr):
 rows=[]
 for i,name in enumerate(names):
  scores,_=metrics(arr['generated'][i:i+1],arr['reference'][i:i+1],arr['sequence_ids'][i:i+1],arr['atom_mask'][i:i+1])
  bonds=np.linalg.norm(arr['generated'][i,-64:,:-1,2]-arr['generated'][i,-64:,1:,0],axis=-1)
  rows.append(dict(molecule=name,late64_bond_valid=float(((bonds>=1.1)&(bonds<=1.6)).mean()),**scores))
 return rows
ours=per_molecule(a);ref=p/'reports/convergence-snapshot-65000-seed43';original=json.loads((ref/'comparison.json').read_text());comparisons={};boot=np.random.default_rng(93342).integers(32,size=(10000,32))
for baseline in ['hierarchical','single-scale']:
 with np.load(ref/baseline/'trajectories.npz') as z:
  assert np.array_equal(z['reference'],a['reference']);base=per_molecule(z)
 assert names==[r['molecule'] for r in original['results'][baseline]['per_molecule']]
 delta={}
 for key in ['late64_bond_valid','bond_valid','velocity_sliced_w1','acf_mae']:
  d=np.array([x[key]-y[key] for x,y in zip(ours,base)]);delta[key]=dict(mean=float(d.mean()),ci95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
 if baseline=='hierarchical':delta['validation_loss_change']=loss['total']-original['results'][baseline]['validation_loss']['total']
 comparisons[baseline]=delta
(out/'comparison.json').write_text(json.dumps(dict(status='complete',repair=arm,checkpoint=str(src),validation_loss=loss,pooled=pooled,per_molecule=ours,paired_differences=comparisons,scope='Development evaluation; seed43 only, paired molecule intervals do not establish seed robustness'),indent=2))
print('REPAIR AUDIT COMPLETE',arm)
