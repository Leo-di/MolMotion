"""Atom14 trajectory containers; coordinates in Angstrom, time in ps."""
from pathlib import Path
import json
import numpy as np

class TrajectoryBatch:
    def __init__(self,coordinates,sequence,atom_mask,frame_interval_ps=10.,metadata=None):
        self.coordinates=np.asarray(coordinates);self.sequence=np.asarray(sequence);self.atom_mask=np.asarray(atom_mask)
        self.frame_interval_ps=frame_interval_ps;self.metadata=metadata or {}

    def save_npz(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,generated=self.coordinates,sequence_ids=np.repeat(self.sequence[None],len(self.coordinates),axis=0),atom_mask=np.repeat(self.atom_mask[None],len(self.coordinates),axis=0),frame_interval_ps=self.frame_interval_ps)
        p.with_suffix('.json').write_text(json.dumps(self.metadata,indent=2))

    def bond_validity(self):
        bonds=np.linalg.norm(self.coordinates[:,:,:-1,2]-self.coordinates[:,:,1:,0],axis=-1)
        return float(((bonds>=1.1)&(bonds<=1.6)).mean())

    def evaluate(self,reference):
        from .metrics import metrics
        ref=np.asarray(reference)
        if ref.shape!=self.coordinates.shape:raise ValueError('Reference must match [samples,frames,4,14,3]')
        if not np.isfinite(ref).all():raise ValueError('Nonfinite reference')
        return metrics(self.coordinates,ref,np.repeat(self.sequence[None],len(ref),axis=0),np.repeat(self.atom_mask[None],len(ref),axis=0))[0]

    def _mdtraj(self,sample):
        import mdtraj as md
        from ._vendor.mdgen import residue_constants as rc
        if not 0<=sample<len(self.coordinates):raise ValueError('Invalid sample index')
        top=md.Topology();chain=top.add_chain();atoms=[];indices=[];atom_by_res=[]
        for i,aa in enumerate(self.sequence):
            resname=rc.restype_1to3[rc.restypes[int(aa)]];res=top.add_residue(resname,chain);current={}
            for j,name in enumerate(rc.restype_name_to_atom14_names[resname]):
                if name and self.atom_mask[i,j]:
                    atom=top.add_atom(name,md.element.get_by_symbol(name[0]),res)
                    atoms.append(atom);indices.append((i,j));current[name]=atom
            atom_by_res.append(current)
        top.create_standard_bonds()
        # MDTraj coordinates are nm, while the model coordinates are Angstrom.
        coords=np.stack([self.coordinates[sample,:,i,j] for i,j in indices],axis=1)/10.
        return md.Trajectory(coords,top,time=np.arange(len(coords))*self.frame_interval_ps)

    def save_pdb(self,path,sample=0):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);self._mdtraj(sample).save_pdb(str(p))

    def save_xtc(self,path,sample=0):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);self._mdtraj(sample).save_xtc(str(p))

    def propose_seeds(self,count=3):
        """Screen geometry and greedily diversify torsions; not an MD stability guarantee."""
        if count<1:raise ValueError('count must be positive')
        from .metrics import angles
        x=self.coordinates[:,1:].reshape(-1,4,14,3)
        bond=np.linalg.norm(x[:,:-1,2]-x[:,1:,0],axis=-1)
        valid=((bond>=1.1)&(bond<=1.6)).all(-1)
        for a,b in [(0,2),(0,3),(1,3)]:
            dist=np.linalg.norm(x[:,a,:,None]-x[:,b,None,:],axis=-1)
            present=self.atom_mask[a,:,None]*self.atom_mask[b,None,:]
            valid&=~((dist<1.2)&present.astype(bool)).any((1,2))
        candidates=np.flatnonzero(valid)
        if len(candidates)<count:raise ValueError(f'Only {len(candidates)} frames passed geometry screening')
        phi,psi=angles(x[None]);theta=np.concatenate((phi[0],psi[0]),axis=-1)
        p0,q0=angles(self.coordinates[:1,:1]);origin=np.concatenate((p0[0,0],q0[0,0]))
        scores=(1-np.cos(theta[candidates]-origin)).mean(-1);chosen=[]
        for _ in range(count):
            k=int(np.argmax(scores));idx=int(candidates[k]);chosen.append(idx)
            scores=np.minimum(scores,(1-np.cos(theta[candidates]-theta[idx])).mean(-1));scores[k]=-np.inf
        t=self.coordinates.shape[1]-1
        return [dict(sample=i//t,frame=i%t+1,time_ps=(i%t+1)*self.frame_interval_ps,coordinates=x[i].copy()) for i in chosen]
