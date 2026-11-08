import numpy as np

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

