import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from common_v2 import prepare_batch,get_offsets,decode_latents,true_atom14,atom14_mask
from model import base_noise

def tensors(batch,model):
    prepared=prepare_batch(batch);raw=prepared['latents'].float()
    z=(raw-model.mean)/model.scale
    initial_frames=prepared['rigids'][:,0]
    geom=initial_frames[:,0:1].invert().compose(initial_frames).to_tensor_7().clone()
    geom[...,:4]*=torch.where(geom[...,:1]<0,-1.,1.)
    geom[...,4:]/=5.
    mask=prepared['loss_mask'][:,0].float()
    z=z*mask[:,None]
    return prepared,z,geom,mask

@torch.no_grad()
def fit_normalization(dataset,frames=257,clips=512):
    total=torch.zeros(21,dtype=torch.float64);sq=total.clone();count=total.clone()
    residual_sq=total.clone();residual_count=total.clone()
    for i in range(0,clips,16):
        batch,_=dataset.batch(min(16,clips-i),frames,300000+i,balanced=True,name_offset=i)
        prepared=prepare_batch(batch);x=prepared['latents'].double();mask=prepared['loss_mask'].double()
        total+=(x*mask).sum((0,1,2));sq+=(x.square()*mask).sum((0,1,2));count+=mask.sum((0,1,2))
        future=x[:,1:];blocks=future.reshape(len(x),-1,8,4,21)
        residual=(blocks-blocks.mean(2,keepdim=True)).reshape_as(future)
        residual_sq+=(residual.square()*mask[:,1:]).sum((0,1,2));residual_count+=mask[:,1:].sum((0,1,2))
    mean=total/count.clamp_min(1)
    scale=(sq/count.clamp_min(1)-mean.square()).clamp_min(.01).sqrt()
    rscale=(residual_sq/residual_count.clamp_min(1)).sqrt()/scale
    rscale=rscale.clamp_min(.1)
    return dict(mean=mean.float(),scale=scale.float(),residual_scale=rscale.float(),
                training_names=dataset.names,clips=clips,frames=frames,test_loaded=False)

def masked_mse(x,y,mask):
    m=mask[:,None].expand_as(x)
    return ((x-y).square()*m).sum()/m.sum().clamp_min(1)

def geometry(raw,batch,prepared):
    xyz,_=decode_latents(raw.float(),prepared,batch['seqres'])
    truth=true_atom14(batch,prepared)
    bonds=torch.linalg.vector_norm(xyz[:,1:,:-1,2]-xyz[:,1:,1:,0],dim=-1)
    target=torch.linalg.vector_norm(truth[:,1:,:-1,2]-truth[:,1:,1:,0],dim=-1)
    bond=F.smooth_l1_loss(bonds,target)
    mask=atom14_mask(batch['seqres'],raw.shape[1])[:,1:]
    numerator=raw.new_zeros(());denominator=raw.new_zeros(())
    for a,b in [(0,2),(0,3),(1,3)]:
        distance=torch.cdist(xyz[:,1:,a],xyz[:,1:,b])
        valid=mask[:,:,a,:,None]*mask[:,:,b,None,:]
        numerator+=(F.relu(1.2-distance).square()*valid).sum();denominator+=valid.sum()
    clash=numerator/denominator.clamp_min(1)
    norm=(raw[:,1:,:,7:].reshape(*raw[:,1:].shape[:3],7,2).norm(dim=-1)-1).square()
    torsion_mask=batch['torsion_mask'][:,None].expand_as(norm)
    torsion=(norm*torsion_mask).sum()/torsion_mask.sum().clamp_min(1)
    return bond,clash,torsion

def loss_terms(wrapped,batch,progress=0,validation=False):
    model=wrapped.module if hasattr(wrapped,'module') else wrapped
    prepared,z,geom,mask=tensors(batch,model)
    future=z[:,1:];b,t,r,d=future.shape;stride=model.stride
    slow=future.reshape(b,t//stride,stride,r,d).mean(2)
    residual=(future-slow.repeat_interleave(stride,1))/model.residual_scale
    previous=torch.cat([z[:,:1],slow[:,:-1]],1)
    ns=base_noise(slow.shape,z.device)*mask[:,None]
    nf=model.project(base_noise(residual.shape,z.device,True))/(1-1/stride)**.5
    nf=nf*mask[:,None]
    ts=torch.rand(b,device=z.device)*.98+.01;tf=ts if getattr(model,"single_scale",False) else torch.rand(b,device=z.device)*.98+.01
    st=ts[:,None,None,None];ft=tf[:,None,None,None]
    sx=ns*(1-st)+slow*st;fx=nf*(1-ft)+residual*ft
    if getattr(model,'single_scale',False):
        joint=sx.repeat_interleave(stride,1)+fx*model.residual_scale
        velocity=wrapped(z[:,0],geom,batch['seqres'],joint,ts)
        vs=velocity.reshape(b,t//stride,stride,r,d).mean(2)
        vf=model.project(velocity)/model.residual_scale
    else:
        vs,vf=wrapped(z[:,0],geom,batch['seqres'],sx,previous,ts,fx,slow,tf)
    vs=vs*mask[:,None];vf=vf*mask[:,None]
    slow_fm=masked_mse(vs,slow-ns,mask);fast_fm=masked_mse(vf,residual-nf,mask)
    shat=sx+(1-st)*vs;fhat=fx+(1-ft)*vf
    geometry_slow=shat.detach() if (not validation and getattr(model,"repair","none")=="geometry_detach_slow") else shat
    endpoint=geometry_slow.repeat_interleave(stride,1)+fhat*model.residual_scale
    norm_raw=torch.cat([z[:,:1],endpoint],1)*model.scale+model.mean
    norm_raw[:,0]=prepared['latents'][:,0]
    # Decode and evaluate geometry in FP32 even under mixed precision.
    with torch.autocast(device_type='cuda',enabled=False):
        bond,clash,torsion=geometry(norm_raw,batch,prepared)
    geom_weight=1. if validation else min(1.,progress/2000)
    total=slow_fm+fast_fm+geom_weight*(bond+.1*clash+.01*torsion)
    details=dict(total=total,slow_fm=slow_fm,fast_fm=fast_fm,bond=bond,clash=clash,torsion=torsion,
        slow_endpoint_mse=masked_mse(shat,slow,mask),fast_endpoint_mse=masked_mse(fhat,residual,mask))
    regression={}
    for name,pred,target in [('slow',vs,slow-ns),('fast',vf,residual-nf)]:
        valid=mask[:,None].expand_as(pred).bool();x=pred[valid];y=target[valid]
        regression[name+'_mae']=(x-y).abs().mean()
        regression[name+'_r2']=1-(x-y).square().sum()/(y-y.mean()).square().sum().clamp_min(1e-8)
    aux=dict(prediction=endpoint.detach(),target=future.detach(),prepared=prepared,
             scatter_prediction=vf.detach()[mask[:,None].expand_as(vf).bool()][:8192],
             scatter_target=(residual-nf).detach()[mask[:,None].expand_as(vf).bool()][:8192])
    return total,details,regression,aux
