import json,sys,os
from pathlib import Path
import numpy as np
import torch
from common_v2 import Trajectories
from model import make_model
from evaluation import free_validation,metrics
from figures import validation_figures,trajectory_video
P=Path(os.environ['V2_ROOT']);base=P.parent;seed=int(sys.argv[1]);budget=int(sys.argv[2]);out=P/'reports'/f'fixed-{budget}-seed{seed}';out.mkdir(parents=True,exist_ok=True)
records={};cohorts={}
for arm in ['hierarchical','single-scale']:
 source=(base/'queued-full-20261106-v1/runs'/f'{arm}-seed{seed}' if budget==100000 else P/'runs'/f'{arm}-seed{seed}-150k')/f'checkpoint-step-{budget:07d}.pt'
 saved=torch.load(source,map_location='cpu',weights_only=False);assert saved['step']==budget
 cfg=saved['config'];model=make_model(arm,cfg['width'],cfg['slow_layers'],cfg['fast_layers'],cfg['heads']).cuda();model.load_state_dict(saved['model']);model.load_state_dict(saved.get('ema') or saved['model'],strict=False);model.joint_ode=cfg.get('joint_ode',64);del saved
 dataset=Trajectories('val','cuda');dest=out/arm
 pooled,arrays=free_validation(model,dataset,256,dest,budget,32,16,32)
 validation_figures(dest,arrays);trajectory_video(dest,arrays)
 metadata=json.loads((dest/'metrics.json').read_text())['sample_info'];names=[n for row in metadata for n in row['names']]
 per=[]
 for i,n in enumerate(names):
  score,_=metrics(arrays['generated'][i:i+1],arrays['reference'][i:i+1],arrays['sequence_ids'][i:i+1],arrays['atom_mask'][i:i+1]);per.append(dict(molecule=n,**score))
 records[arm]=dict(step=budget,checkpoint=str(source),pooled=pooled,per_molecule=per);cohorts[arm]=arrays['reference'];del model,dataset;torch.cuda.empty_cache()
assert np.array_equal(cohorts['hierarchical'],cohorts['single-scale'])
if seed==43 and budget==100000:
 source=P/'mdgen/reports/step-100000';m=json.loads((source/'results.json').read_text());assert m['training_step']==budget
 with np.load(source/'trajectories.npz') as z:assert np.max(np.abs(z['reference']-cohorts['hierarchical']))<5e-5
 records['mdgen']=dict(step=budget,checkpoint=m['checkpoint'],pooled=m['pooled'],per_molecule=m['per_molecule'])
boot=np.random.default_rng(93342).integers(32,size=(10000,32));paired={}
for arm in records:
 if arm=='hierarchical':continue
 assert [x['molecule'] for x in records[arm]['per_molecule']]==[x['molecule'] for x in records['hierarchical']['per_molecule']]
 paired[arm]={}
 for key in ['bond_valid','acf_mae','velocity_sliced_w1','motion_ratio','phi_psi_js']:
  dif=np.array([x[key]-y[key] for x,y in zip(records['hierarchical']['per_molecule'],records[arm]['per_molecule'])]);paired[arm][key]=dict(mean=float(dif.mean()),ci95=np.quantile(dif[boot].mean(1),[.025,.975]).tolist())
(out/'comparison.json').write_text(json.dumps(dict(status='complete',budget=budget,seed=seed,results=records,hierarchical_minus=paired,scope='Same training update and sample budget; fixed development clips and noise seeds. Model sizes, FLOPs and solvers differ. No test-set selection.'),indent=2))
print('AUDIT COMPLETE',out)
