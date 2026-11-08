import json
import time
from pathlib import Path
import numpy as np
import torch
from common_v2 import decode_latents,true_atom14,atom14_mask
from dynamics import tensors
from model import generate

def dihedral(a,b,c,d):
    axis=c-b;axis/=np.linalg.norm(axis,axis=-1,keepdims=True).clip(1e-8)
    v=a-b;v-=np.sum(v*axis,axis=-1,keepdims=True)*axis
    w=d-c;w-=np.sum(w*axis,axis=-1,keepdims=True)*axis
    return np.arctan2(np.sum(np.cross(axis,v)*w,-1),np.sum(v*w,-1))

def angles(x):
    phi=[];psi=[]
    for i in [1,2]:
        phi.append(dihedral(x[:,:,i-1,2].copy(),x[:,:,i,0],x[:,:,i,1],x[:,:,i,2]))
        psi.append(dihedral(x[:,:,i,0].copy(),x[:,:,i,1],x[:,:,i,2],x[:,:,i+1,0]))
    return np.stack(phi,-1),np.stack(psi,-1)

def distribution(x):
    phi,psi=angles(x)
    hist=np.histogram2d(phi.ravel(),psi.ravel(),bins=32,range=[[-np.pi,np.pi]]*2)[0]
    hist=(hist+1e-9)/(hist.sum()+1e-9*hist.size)
    acf=[]
    for lag in [1,2,4,8,16,32,64]:
        if lag>=x.shape[1]:break
        acf.append([lag,float((np.cos(phi[:,lag:]-phi[:,:-lag]).mean()+np.cos(psi[:,lag:]-psi[:,:-lag]).mean())/2)])
    vel=np.diff(x[:,:,:,:4],axis=1).reshape(len(x),x.shape[1]-1,48)
    power=(np.abs(np.fft.rfft(vel,axis=1))**2).mean((0,2))
    power/=max(power.sum(),1e-12)
    distance=np.linalg.norm(x[:,:,:,None,1]-x[:,:,None,:,1],axis=-1)
    contact=(distance<8).mean((0,1))
    v=vel.ravel();z=(v-v.mean())/max(v.std(),1e-8)
    return dict(phi_psi_histogram=hist,acf=np.array(acf),velocity_psd=power,
                ca_contacts=contact,velocity_excess_kurtosis=np.mean(z**4)-3,velocities=vel)

def metrics(prediction,reference,sequence,atom_mask):
    g=distribution(prediction);r=distribution(reference)
    mask=atom_mask[:,None]
    squared=((prediction-reference)**2).sum(-1)
    coordinate=np.sqrt((squared*mask).sum((-1,-2))/mask.sum((-1,-2)).clip(1)).mean()
    bonds=np.linalg.norm(prediction[:,:,:-1,2]-prediction[:,:,1:,0],axis=-1)
    bond_valid=((bonds>=1.1)&(bonds<=1.6))
    pstep=np.sqrt(((np.diff(prediction,axis=1)**2).sum(-1)*mask).sum((-1,-2))/mask.sum((-1,-2)).clip(1)).mean()
    rstep=np.sqrt(((np.diff(reference,axis=1)**2).sum(-1)*mask).sum((-1,-2))/mask.sum((-1,-2)).clip(1)).mean()
    clashes=np.zeros(prediction.shape[:2],dtype=bool)
    for a,b in [(0,2),(0,3),(1,3)]:
        dist=np.linalg.norm(prediction[:,:,a,:,None]-prediction[:,:,b,None,:],axis=-1)
        valid=atom_mask[:,a,:,None]*atom_mask[:,b,None,:]
        clashes|=((dist<1.2)&valid[:,None].astype(bool)).any((-1,-2))
    m=(g['phi_psi_histogram']+r['phi_psi_histogram'])/2
    js=.5*np.sum(g['phi_psi_histogram']*np.log(g['phi_psi_histogram']/m))+.5*np.sum(r['phi_psi_histogram']*np.log(r['phi_psi_histogram']/m))
    left=g['velocities'].reshape(-1,48);right=r['velocities'].reshape(-1,48)
    rng=np.random.default_rng(831);n=min(4096,len(left),len(right))
    left=left[rng.choice(len(left),n,replace=False)];right=right[rng.choice(len(right),n,replace=False)]
    directions=rng.normal(size=(48,128));directions/=np.linalg.norm(directions,axis=0)
    sw1=np.abs(np.sort(left@directions,axis=0)-np.sort(right@directions,axis=0)).mean()
    numbers=dict(stochastic_coordinate_rmsd=float(coordinate),bond_valid=float(bond_valid.mean()),
        bond_mean=float(bonds.mean()),bond_p95=float(np.quantile(bonds,.95)),
        nonadjacent_clash_frame_fraction=float(clashes.mean()),generated_step_rmsd=float(pstep),
        reference_step_rmsd=float(rstep),motion_ratio=float(pstep/max(rstep,1e-8)),
        phi_psi_js=float(js),acf_mae=float(np.abs(g['acf'][:,1]-r['acf'][:,1]).mean()),
        velocity_sliced_w1=float(sw1),velocity_psd_l1=float(np.abs(g['velocity_psd']-r['velocity_psd']).sum()),
        velocity_excess_kurtosis=float(g['velocity_excess_kurtosis']),reference_velocity_excess_kurtosis=float(r['velocity_excess_kurtosis']),
        first_condition_rmsd=float(np.sqrt(squared[:,0].mean())))
    arrays=dict(generated_phi_psi=g['phi_psi_histogram'],reference_phi_psi=r['phi_psi_histogram'],
        generated_acf=g['acf'],reference_acf=r['acf'],generated_psd=g['velocity_psd'],reference_psd=r['velocity_psd'],
        generated_contacts=g['ca_contacts'],reference_contacts=r['ca_contacts'],
        bond_valid_by_time=bond_valid.mean(0),ca_error_by_time=np.sqrt(squared[:,:,:,1]).mean(0))
    return numbers,arrays

@torch.no_grad()
def free_validation(model,dataset,future,output,step,clips=8,slow_steps=16,fast_steps=32):
    generated=[];reference=[];sequences=[];mask=[];slow_values=[];residual_values=[];true_slow=[];true_residual=[];info=[]
    count=max(2,clips);sampling_seconds=0.
    for offset in range(0,count,2):
        batch,record=dataset.batch(min(2,count-offset),future+1,91000+offset,balanced=True,name_offset=offset)
        prepared,z,geom,feature_mask=tensors(batch,model)
        torch.cuda.synchronize();sampling_begun=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            values,slow,residual=generate(model,z[:,0],geom,batch['seqres'],feature_mask,future,40100+offset,slow_steps,fast_steps)
        torch.cuda.synchronize();sampling_seconds+=time.perf_counter()-sampling_begun
        raw=values*model.scale+model.mean;raw[:,0]=prepared['latents'][:,0]
        xyz,_=decode_latents(raw,prepared,batch['seqres'])
        truth=true_atom14(batch,prepared)
        generated.append(xyz.cpu().numpy());reference.append(truth.cpu().numpy());sequences.append(batch['seqres'].cpu().numpy())
        mask.append(atom14_mask(batch['seqres'],future+1)[:,0].cpu().numpy())
        slow_values.append(slow.cpu().numpy());residual_values.append(residual.cpu().numpy());info.append(record)
        target=z[:,1:];blocks=target.reshape(len(z),future//model.stride,model.stride,4,21)
        coarse=blocks.mean(2)
        true_slow.append(coarse.cpu().numpy())
        true_residual.append(((target-coarse.repeat_interleave(model.stride,1))/model.residual_scale).cpu().numpy())
    prediction=np.concatenate(generated);truth=np.concatenate(reference);seq=np.concatenate(sequences);am=np.concatenate(mask)
    assert np.isfinite(prediction).all()
    numbers,arrays=metrics(prediction,truth,seq,am)
    numbers.update(sampling_seconds=sampling_seconds,generated_clips=count)
    arrays.update(generated=prediction,reference=truth,sequence_ids=seq,atom_mask=am,
                  generated_slow=np.concatenate(slow_values),generated_residual=np.concatenate(residual_values),
                  reference_slow=np.concatenate(true_slow),reference_residual=np.concatenate(true_residual))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'trajectories.npz',**arrays)
    (output/'metrics.json').write_text(json.dumps(dict(step=step,metrics=numbers,sample_info=info,architecture='single-scale' if getattr(model,'single_scale',False) else 'hierarchical',
        generation='joint single-scale flow from known initial state only' if getattr(model,'single_scale',False) else 'free causal coarse rollout plus conditional stochastic residual; reference future never used as input',
        frame_interval_ps=10,test_loaded=False),indent=2))
    return numbers,arrays
