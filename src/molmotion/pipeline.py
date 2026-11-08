"""Four-residue PDB -> slow-fast model -> atom14 trajectories."""
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

MODEL_URL = 'https://github.com/Leo-di/MolMotion/releases/download/v0.1.0'


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def load_pdb(path):
    import mdtraj as md
    from ._vendor.mdgen import residue_constants as rc
    traj=md.load_pdb(str(path))
    residues=[r for r in traj.topology.residues if r.name in rc.restype_3to1]
    other=[r.name for r in traj.topology.residues if r.name not in rc.restype_3to1 and r.name not in {'HOH','WAT'}]
    if len(residues)!=4 or len({r.chain.index for r in residues})!=1 or other:
        raise ValueError('Expected one chain of four standard amino acids (water may be ignored); no caps, ligands or modified residues')
    xyz=np.zeros((4,14,3),dtype=np.float32);mask=np.zeros((4,14),dtype=np.float32);seq=[]
    for i,residue in enumerate(residues):
        seq.append(rc.restype_order[rc.restype_3to1[residue.name]])
        names=rc.restype_name_to_atom14_names[residue.name]
        seen=set()
        for atom in residue.atoms:
            if atom.name in names:
                if atom.name in seen:raise ValueError('Duplicate atom/alternate location: '+atom.name)
                seen.add(atom.name);j=names.index(atom.name)
                xyz[i,j]=traj.xyz[0,atom.index]*10.;mask[i,j]=1
        missing=[name for name in names if name and name not in seen]
        if missing:raise ValueError(f'Missing heavy atoms in residue {i+1}: {missing}')
    if not np.isfinite(xyz).all():raise ValueError('Nonfinite input coordinates')
    bonds=np.linalg.norm(xyz[:-1,2]-xyz[1:,0],axis=-1)
    if ((bonds<.8)|(bonds>2.2)).any():
        raise ValueError('Initial peptide bonds are disconnected; unwrap periodic coordinates and check topology')
    return xyz,np.array(seq,dtype=np.int64),mask


def encode_initial(xyz,sequence,device):
    import torch
    from ._vendor.mdgen.geometry import atom14_to_frames,atom14_to_atom37,atom37_to_torsions
    positions=torch.as_tensor(xyz[None],device="cpu",dtype=torch.float32)
    seq=torch.as_tensor(sequence[None],device="cpu",dtype=torch.long)
    frames=atom14_to_frames(positions)
    atom37=atom14_to_atom37(positions,seq)
    torsions,tmask=atom37_to_torsions(atom37,seq)
    offsets=frames.invert().compose(frames).to_tensor_7()
    offsets[...,:4]*=torch.where(offsets[...,:1]<0,-1.,1.)
    raw=torch.cat((offsets,torsions.reshape(1,4,14)),dim=-1)
    geom=frames[:,0:1].invert().compose(frames).to_tensor_7().clone()
    geom[...,:4]*=torch.where(geom[...,:1]<0,-1.,1.);geom[...,4:]/=5.
    feature_mask=torch.cat((torch.ones_like(offsets),tmask[...,None].expand(-1,-1,-1,2).reshape(1,4,14)),dim=-1)
    return raw.to(device),geom.to(device),seq.to(device),feature_mask.to(device),frames.map_tensor_fn(lambda value: value.to(device))


class MolMotionPipeline:
    def __init__(self,model,config,device='cpu'):
        self.model=model.to(device).eval();self.config=config;self.device=str(device)

    @classmethod
    def from_pretrained(cls,model_path='tetrapeptide-v0.1.0',device='cpu',cache_dir=None):
        import torch
        from safetensors.torch import load_file
        from .model import make_model
        if model_path=='tetrapeptide-v0.1.0':
            root=Path(cache_dir or Path.home()/'.cache/molmotion')/'tetrapeptide-v0.1.0'
            root.mkdir(parents=True,exist_ok=True)
            for name in ['config.json','model.safetensors']:
                target=root/name
                if not target.exists():
                    partial=target.with_suffix(target.suffix+'.partial')
                    urllib.request.urlretrieve(MODEL_URL+'/'+name,partial)
                    partial.replace(target)
        else:root=Path(model_path)
        c=json.loads((root/'config.json').read_text())
        if c.get('format_version')!=1 or c.get('feature_version')!='mdgen-offset21-v1' or c.get('stride')!=8:
            raise ValueError('Unsupported model or feature format')
        if c.get('architecture') not in {'hierarchical','single-scale'}:raise ValueError('Unknown architecture')
        if c.get('frame_interval_ps')!=10.:raise ValueError('Unsupported frame interval')
        if any(not isinstance(c.get(k),int) or c[k]<=0 for k in ['width','slow_layers','fast_layers','heads']):raise ValueError('Invalid architecture dimensions')
        weights=root/c['weights']
        if weights.resolve().parent!=root.resolve():raise ValueError('Weights must be inside the model directory')
        if _sha(weights)!=c['weights_sha256']:raise ValueError('Weight SHA256 mismatch; remove corrupt cache and retry')
        model=make_model(c['architecture'],c['width'],c['slow_layers'],c['fast_layers'],c['heads'])
        model.load_state_dict(load_file(str(weights),device='cpu'),strict=True)
        for key in ['mean','scale','residual_scale']:
            v=getattr(model,key)
            if not torch.isfinite(v).all() or (key!='mean' and (v<=0).any()):raise ValueError('Invalid normalization')
        return cls(model,c,device)

    def generate(self,pdb_file,num_frames=256,num_samples=1,seed=42,slow_steps=16,fast_steps=32):
        import torch
        from .model import generate
        from .trajectory import TrajectoryBatch
        from ._vendor.mdgen.geometry import frames_torsions_to_atom14
        from ._vendor.mdgen.rigid_utils import Rigid
        if num_frames!=self.config['future_frames']:
            raise ValueError('This release is validated for the configured horizon only')
        if num_samples<1 or slow_steps<1 or fast_steps<1:raise ValueError('Samples and integration steps must be positive')
        xyz,sequence,mask=load_pdb(pdb_file)
        raw,geom,seq,feature_mask,frames=encode_initial(xyz,sequence,self.device)
        initial=(raw-self.model.mean)/self.model.scale*feature_mask
        output=[]
        with torch.inference_mode():
            for index in range(num_samples):
                values,_,_=generate(self.model,initial,geom,seq,feature_mask,num_frames,seed+index,slow_steps,fast_steps)
                latents=values*self.model.scale+self.model.mean;latents[:,0]=raw
                offsets=latents[...,:7]
                future_frames=frames[:,None].compose(Rigid.from_tensor_7(offsets,normalize_quats=True))
                torsions=latents[...,7:].reshape(1,num_frames+1,4,7,2)
                torsions=torsions/torch.linalg.vector_norm(torsions,dim=-1,keepdim=True).clamp_min(1e-8)
                coords=frames_torsions_to_atom14(future_frames,torsions,seq[:,None].expand(1,num_frames+1,4))
                arr=coords[0].cpu().numpy();arr[0]=xyz
                if not np.isfinite(arr).all():raise RuntimeError('Nonfinite model output')
                output.append(arr)
        metadata=dict(model_config=self.config,seed=seed,slow_steps=slow_steps,fast_steps=fast_steps,initial_frame='original PDB coordinates',units='angstrom',frame_interval_ps=self.config['frame_interval_ps'])
        return TrajectoryBatch(np.stack(output),sequence,mask,self.config['frame_interval_ps'],metadata)
