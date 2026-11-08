import torch
import dynamics
from model import make_model

torch.set_num_threads(2);torch.manual_seed(7)
m=make_model('hierarchical',32,1,1,4)
z=torch.randn(1,17,4,21);geom=torch.randn(1,4,7);mask=torch.ones(1,4,21)
batch={'seqres':torch.zeros(1,4,dtype=torch.long)}
dynamics.tensors=lambda batch,model: ({'latents':z},z,geom,mask)
def surrogate(raw,batch,prepared):
 q=raw[:,1:].square().mean();return q,q*.2,q*.1
dynamics.geometry=surrogate
params=list(m.slow.parameters())
def calc(repair,validation=False):
 m.repair=repair;torch.manual_seed(888)
 loss,n,_,_=dynamics.loss_terms(m,batch,26000,validation)
 grad=torch.autograd.grad(n['bond']+.1*n['clash']+.01*n['torsion'],params,allow_unused=True)
 norm=sum(float(g.square().sum()) for g in grad if g is not None)
 return float(loss),norm
original=calc('none');detached=calc('geometry_detach_slow');val=calc('geometry_detach_slow',True)
assert abs(original[0]-detached[0])<1e-6
assert original[1]>0 and detached[1]==0 and val[1]>0
# Dropout must disappear at evaluation and be active during training.
a=m.slow.blocks[0].time;x=torch.randn(2,8,32)
a.eval();y=a(x);a.dropout_p=.1;assert torch.equal(y,a(x))
a.train();torch.manual_seed(3);y=a(x);torch.manual_seed(4);assert not torch.equal(y,a(x))
print({'passed':True,'geometry_gradient_norm_squared':{'original':original[1],'detached':detached[1],'validation':val[1]},'same_forward_loss':True,'scope':'Small synthetic unit test of gradient routing and dropout; not a scientific result'})
