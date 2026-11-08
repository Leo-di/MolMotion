import sys,os,json,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'mdgen'))
import baseline as b
from common_v2 import Trajectories,prepare_batch
from evaluation import metrics
from figures import trajectory_video
step=int(sys.argv[1]);out=P/'mdgen/reports'/f'step-{step}';run=P/'mdgen/runs/matched-seed43'
saved=b.checkpoint(run/'last.pt');assert saved['step']==step
model,transport=b.model_and_transport(b.upstream_config(saved['upstream']),'cuda');model.load_state_dict(saved['ema']);model.eval();del saved
val=Trajectories('val','cuda');loss=[]
with torch.no_grad():
 for offset in range(0,32,2):
  batch,_=val.batch(2,257,81000+offset,balanced=True,name_offset=offset);p=prepare_batch(batch);torch.manual_seed(71000+offset)
  loss.append(float(transport.training_losses(model,p['latents'],mask=p['loss_mask'],model_kwargs=b.conditioning(p))['loss'].mean()))
value=float(np.mean(loss));assert np.isfinite(value)
with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=value,scope='fixed 32-molecule development clips; MDGen objective only'))+'\n')
del val,model,transport;torch.cuda.empty_cache()
with np.load(out/'trajectories.npz') as z:arrays={k:z[k] for k in z.files}
numbers,more=metrics(arrays['generated'],arrays['reference'],arrays['sequence_ids'],arrays['atom_mask']);arrays.update(more)
np.savez_compressed(out/'diagnostic-arrays.npz',**arrays)
f,aa=plt.subplots(2,2,figsize=(12,8))
for a,k,t in zip(aa[0],['reference_phi_psi','generated_phi_psi'],['Reference phi/psi','MDGen phi/psi']):
 im=a.imshow(arrays[k].T,origin='lower',extent=[-180,180,-180,180],cmap='magma',aspect='auto');a.set_title(t);f.colorbar(im,ax=a)
aa[1,0].plot(arrays['bond_valid_by_time']);aa[1,0].set(xlabel='Frame',ylabel='Bond validity')
for k in ['reference_acf','generated_acf']:aa[1,1].plot(arrays[k][:,0],arrays[k][:,1],label=k)
aa[1,1].legend();aa[1,1].set_title('Temporal autocorrelation');f.tight_layout();f.savefig(out/'geometry-motion-heatmaps.png',dpi=180);plt.close(f)
trajectory_video(out,arrays)
records=[json.loads(l) for l in (run/'train.jsonl').read_text().splitlines()];vals=[json.loads(l) for l in (run/'validation.jsonl').read_text().splitlines()]
f,aa=plt.subplots(1,2,figsize=(12,4));aa[0].plot([r['step'] for r in records],[r['loss'] for r in records]);aa[0].set_title('Training objective');aa[1].plot([r['step'] for r in vals],[r['loss'] for r in vals],'-o');aa[1].set_title('Fixed development objective');f.tight_layout();f.savefig(run/'training-validation.png',dpi=180);plt.close(f)
if value<=min(r['loss'] for r in vals):shutil.copyfile(run/'last.pt',run/'best-validation.pt')
print(json.dumps(dict(step=step,validation_loss=value,artifacts=str(out))))
