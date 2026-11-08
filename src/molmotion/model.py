"""Hierarchical molecular dynamics: causal coarse flow and conditional residual flow."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def sinusoidal(t,width):
    frequencies=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device)/(width//2-1))
    phase=t[...,None]*frequencies
    return torch.cat([phase.cos(),phase.sin()],-1)

class Attention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.head_dim=width//heads
        self.qkv=nn.Linear(width,3*width);self.output=nn.Linear(width,width)
    def forward(self,x,causal=False):
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=causal)
        return self.output(out.transpose(1,2).reshape(b,n,d))

class Block(nn.Module):
    def __init__(self,width,heads,causal):
        super().__init__();self.causal=causal
        self.time=Attention(width,heads);self.space=Attention(width,heads)
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.modulate=nn.Sequential(nn.SiLU(),nn.Linear(width,2*width))
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(approximate='tanh'),nn.Linear(4*width,width))
    def forward(self,x,condition):
        b,t,r,d=x.shape
        scale,shift=self.modulate(condition).chunk(2,-1)
        v=self.norms[0](x)*(1+.1*scale[:,None,None])+.1*shift[:,None,None]
        v=v.permute(0,2,1,3).reshape(b*r,t,d)
        x=x+self.time(v,self.causal).reshape(b,r,t,d).permute(0,2,1,3)
        x=x+self.space(self.norms[1](x).reshape(b*t,r,d)).reshape(b,t,r,d)
        return x+self.ffn(self.norms[2](x))

class FlowTransformer(nn.Module):
    def __init__(self,width,layers,heads,causal,checkpointing=True):
        super().__init__();self.width=width;self.checkpointing=checkpointing
        self.input=nn.Linear(21,width)
        self.context=nn.Sequential(nn.Linear(21,width),nn.SiLU(),nn.Linear(width,width))
        self.initial=nn.Sequential(nn.Linear(28,width),nn.SiLU(),nn.Linear(width,width))
        self.aa=nn.Embedding(21,width);self.residue=nn.Embedding(4,width)
        self.flow_time=nn.Sequential(nn.Linear(width,4*width),nn.SiLU(),nn.Linear(4*width,width))
        self.blocks=nn.ModuleList([Block(width,heads,causal) for _ in range(layers)])
        self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,21))
        nn.init.normal_(self.out[-1].weight,std=.002);nn.init.zeros_(self.out[-1].bias)
    def forward(self,x,t,context,initial,geometry,sequence,positions):
        b,n,r,_=x.shape
        condition=self.flow_time(sinusoidal(t*1000,self.width))
        known=self.initial(torch.cat([initial,geometry],-1))+self.aa(sequence)+self.residue(torch.arange(r,device=x.device))[None]
        values=self.input(x)+self.context(context)+known[:,None]
        values=values+sinusoidal(positions.float(),self.width)[None,:,None]+condition[:,None,None]
        for block in self.blocks:
            if self.training and self.checkpointing:
                values=checkpoint(block,values,condition,use_reentrant=False)
            else:values=block(values,condition)
        return self.out(values).float()

class HierarchicalDynamics(nn.Module):
    def __init__(self,width=768,slow_layers=12,fast_layers=16,heads=12,stride=8,checkpointing=True):
        super().__init__();self.stride=stride
        self.slow=FlowTransformer(width,slow_layers,heads,True,checkpointing)
        self.fast=FlowTransformer(width,fast_layers,heads,False,checkpointing)
        self.register_buffer('mean',torch.zeros(21));self.register_buffer('scale',torch.ones(21))
        self.register_buffer('residual_scale',torch.ones(21))
    def project(self,x):
        b,t,r,d=x.shape
        block=x.reshape(b,t//self.stride,self.stride,r,d)
        return (block-block.mean(2,keepdim=True)).reshape_as(x)
    def forward(self,initial,geometry,sequence,slow_xt,previous_slow,ts,fast_xt,slow_condition,tf):
        slow_positions=(torch.arange(slow_xt.shape[1],device=initial.device)+1)*self.stride
        fast_positions=torch.arange(fast_xt.shape[1],device=initial.device)+1
        vs=self.slow(slow_xt,ts,previous_slow,initial,geometry,sequence,slow_positions)
        condition=slow_condition.repeat_interleave(self.stride,1)
        vf=self.project(self.fast(fast_xt,tf,condition,initial,geometry,sequence,fast_positions))
        return vs,vf

def base_noise(shape,device,student=False):
    x=torch.randn(shape,device=device)
    if student:
        chi=torch.randn((*shape,5),device=device).square().sum(-1).clamp_min(1e-6)
        x=x*torch.sqrt(3/chi)  # unit-variance Student-t with5 degrees of freedom
    return x

@torch.no_grad()
def generate_hierarchical(model,initial,geometry,sequence,feature_mask,future,seed,slow_steps=16,fast_steps=32,student=True):
    """Only known initial features enter this API; no reference future is accepted."""
    model.eval();torch.manual_seed(seed)
    b,r,d=initial.shape;stride=model.stride
    assert future%stride==0
    anchors=[];past_noise=[]
    for index in range(future//stride):
        noise=base_noise((b,1,r,d),initial.device)*feature_mask[:,None]
        value=noise.clone()
        for iteration in range(slow_steps):
            t=torch.full((b,),iteration/slow_steps,device=initial.device)
            if anchors:
                clean_past=torch.cat(anchors,1);npast=torch.cat(past_noise,1)
                noisy_past=npast*(1-t[:,None,None,None])+clean_past*t[:,None,None,None]
                values=torch.cat([noisy_past,value],1)
                previous=torch.cat([initial[:,None],clean_past],1)
            else:
                values=value;previous=initial[:,None]
            pos=(torch.arange(index+1,device=initial.device)+1)*stride
            velocity=model.slow(values,t,previous,initial,geometry,sequence,pos)[:,-1:]*feature_mask[:,None]
            value=value+velocity/slow_steps
        anchors.append(value);past_noise.append(noise)
    slow=torch.cat(anchors,1)
    residual=model.project(base_noise((b,future,r,d),initial.device,student))/math.sqrt(1-1/stride)
    residual=residual*feature_mask[:,None]
    condition=slow.repeat_interleave(stride,1)
    for iteration in range(fast_steps):
        t=torch.full((b,),iteration/fast_steps,device=initial.device)
        positions=torch.arange(future,device=initial.device)+1
        v=model.project(model.fast(residual,t,condition,initial,geometry,sequence,positions))*feature_mask[:,None]
        residual=residual+v/fast_steps
    future_values=condition+residual*model.residual_scale
    return torch.cat([initial[:,None],future_values],1),slow,residual


class SingleScaleDynamics(nn.Module):
    """One304M-parameter flow, no oracle or generated coarse conditioning."""
    single_scale = True

    def __init__(self, width=768, slow_layers=12, fast_layers=16, heads=12,
                 stride=8, checkpointing=True):
        super().__init__()
        self.stride = stride
        self.joint_ode = 64
        self.joint = FlowTransformer(width, slow_layers+fast_layers, heads,
                                     False, checkpointing)
        self.register_buffer('mean', torch.zeros(21))
        self.register_buffer('scale', torch.ones(21))
        self.register_buffer('residual_scale', torch.ones(21))

    def project(self, x):
        b, t, r, d = x.shape
        blocks = x.reshape(b, t//self.stride, self.stride, r, d)
        return (blocks-blocks.mean(2, keepdim=True)).reshape_as(x)

    def forward(self, initial, geometry, sequence, noisy_future, flow_time):
        context = initial[:, None].expand_as(noisy_future)
        positions = torch.arange(noisy_future.shape[1], device=initial.device)+1
        return self.joint(noisy_future, flow_time, context, initial,
                          geometry, sequence, positions)


def make_model(architecture, width=768, slow_layers=12, fast_layers=16, heads=12):
    cls = SingleScaleDynamics if architecture == 'single-scale' else HierarchicalDynamics
    return cls(width, slow_layers, fast_layers, heads)


@torch.no_grad()
def generate(model, initial, geometry, sequence, feature_mask, future, seed,
             slow_steps=16, fast_steps=32, student=True):
    if not getattr(model, 'single_scale', False):
        return generate_hierarchical(model, initial, geometry, sequence,
                                      feature_mask, future, seed, slow_steps,
                                      fast_steps, student)
    model.eval()
    torch.manual_seed(seed)
    b, r, d = initial.shape
    stride = model.stride
    assert future % stride == 0
    coarse_noise = base_noise((b, future//stride, r, d), initial.device)
    residual_noise = model.project(base_noise((b, future, r, d), initial.device, student))
    residual_noise /= math.sqrt(1-1/stride)
    value = (coarse_noise.repeat_interleave(stride, 1)
             + residual_noise*model.residual_scale)*feature_mask[:, None]
    # Same Gaussian-coarse + zero-mean Student residual prior as the hierarchy.
    # Only the factorization/network differs; the single-scale model integrates
    # one joint flow with shared flow time and known initial conditions.
    evaluations = model.joint_ode
    for iteration in range(evaluations):
        t = torch.full((b,), iteration/evaluations, device=initial.device)
        velocity = model(initial, geometry, sequence, value, t)*feature_mask[:, None]
        value = value+velocity/evaluations
    slow = value.reshape(b, future//stride, stride, r, d).mean(2)
    residual = (value-slow.repeat_interleave(stride, 1))/model.residual_scale
    return torch.cat([initial[:, None], value], 1), slow, residual
