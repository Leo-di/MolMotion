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
P=Path(os.environ['V2_ROOT']);base=P.parent;seed=int(sys.argv[1]);budget=int(sys.argv[2]);variant=sys.argv[3];assert variant in ['snapshot','lr02'];out=P/'reports'/f'convergence-{variant}-{budget}-seed{seed}';out.mkdir(parents=True,exist_ok=True)
records={};cohorts={}
for arm in ['hierarchical','single-scale']:
 source=(P/'snapshots'/f'{arm}-seed{seed}' if variant=='snapshot' else P/'runs'/f'{arm}-seed{seed}-lr02-65k')/f'checkpoint-step-{budget:07d}.pt'
 saved=torch.load(source,map_location='cpu',weights_only=False);assert saved['step']==budget
 cfg=saved['config'];model=make_model(arm,cfg['width'],cfg['slow_layers'],cfg['fast_layers'],cfg['heads']).cuda();model.load_state_dict(saved['model']);model.load_state_dict(saved.get('ema') or saved['model'],strict=False);model.joint_ode=cfg.get('joint_ode',64);del saved
 dataset=Trajectories('val','cuda');dest=out/arm
 pooled,arrays=free_validation(model,dataset,256,dest,budget,32,16,32)
 validation_figures(dest,arrays);trajectory_video(dest,arrays)
 loss,_=validate(model,dataset,256,budget)
 metadata=json.loads((dest/'metrics.json').read_text())['sample_info'];names=[n for row in metadata for n in row['names']]
 per=[]
 for i,n in enumerate(names):
  score,_=metrics(arrays['generated'][i:i+1],arrays['reference'][i:i+1],arrays['sequence_ids'][i:i+1],arrays['atom_mask'][i:i+1]);per.append(dict(molecule=n,**score))
 records[arm]=dict(step=budget,checkpoint=str(source),pooled=pooled,validation_loss=loss,per_molecule=per);cohorts[arm]=arrays['reference'];del model,dataset;torch.cuda.empty_cache()
assert np.array_equal(cohorts['hierarchical'],cohorts['single-scale'])
boot=np.random.default_rng(93342).integers(32,size=(10000,32));paired={}
for arm in records:
 if arm=='hierarchical':continue
 assert [x['molecule'] for x in records[arm]['per_molecule']]==[x['molecule'] for x in records['hierarchical']['per_molecule']]
 paired[arm]={}
 for key in ['bond_valid','acf_mae','velocity_sliced_w1','motion_ratio','phi_psi_js']:
  dif=np.array([x[key]-y[key] for x,y in zip(records['hierarchical']['per_molecule'],records[arm]['per_molecule'])]);paired[arm][key]=dict(mean=float(dif.mean()),ci95=np.quantile(dif[boot].mean(1),[.025,.975]).tolist())
(out/'comparison.json').write_text(json.dumps(dict(status='complete',budget=budget,seed=seed,variant=variant,results=records,hierarchical_minus=paired,scope='Same training update and sample budget; fixed development clips and noise seeds. Model sizes, FLOPs and solvers differ. No test-set selection.'),indent=2))
print('AUDIT COMPLETE',out)

if variant=='lr02':
 original=json.loads((P/'reports'/f'convergence-snapshot-{budget}-seed{seed}'/'comparison.json').read_text())['results']
 differences={}
 for arm in records:
  left=records[arm];right=original[arm]
  assert [r['molecule'] for r in left['per_molecule']]==[r['molecule'] for r in right['per_molecule']]
  differences[arm]={'validation_total_change':left['validation_loss']['total']-right['validation_loss']['total']}
  for key in ['bond_valid','velocity_sliced_w1','acf_mae']:
   d=np.array([a[key]-b[key] for a,b in zip(left['per_molecule'],right['per_molecule'])])
   differences[arm][key]={'mean_change':float(d.mean()),'ci95':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
 (out/'low_lr_vs_original.json').write_text(json.dumps(differences,indent=2))
