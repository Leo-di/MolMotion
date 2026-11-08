"""Training diagnostics with raw-data links and fixed-camera trajectory previews."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':1.4,'svg.fonttype':'none','savefig.facecolor':'white'})
BLUE='#0F4D92';RED='#B64342';GREEN='#42949E'

def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)

def read_rows(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows

def curves(run):
    run=Path(run);train=read_rows(run/'train.jsonl');val=read_rows(run/'validation.jsonl');gen=read_rows(run/'generation.jsonl')
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    panels=[(['total','slow_fm','fast_fm'],'Training losses'),(['total','slow_fm','fast_fm'],'Validation losses'),
            (['bond','clash','torsion'],'Geometry losses'),(['lr'],'Learning rate'),
            (['grad_norm'],'Gradient norm'),(['updates_per_second','gpu_peak_gib'],'Throughput / memory')]
    for index,(keys,title) in enumerate(panels):
        ax=axes.flat[index];source=val if index==1 else train
        if index==5:
            rows=[x for x in train if 'updates_per_second' in x]
            twin=ax.twinx()
            ax.plot([x['step'] for x in rows],[x['updates_per_second'] for x in rows],color=BLUE,label='Updates/s')
            twin.plot([x['step'] for x in rows],[x['gpu_peak_gib'] for x in rows],color=RED,label='Peak GiB')
            ax.set(title='Throughput and peak GPU memory',xlabel='Optimizer update',ylabel='Updates/s')
            twin.set_ylabel('GiB');ax.grid(alpha=.15)
            h,l=ax.get_legend_handles_labels();h2,l2=twin.get_legend_handles_labels();ax.legend(h+h2,l+l2,frameon=False,fontsize=9)
            continue
        for key,color in zip(keys,[BLUE,RED,GREEN]):
            rows=[x for x in source if key in x]
            if rows:ax.plot([x['step'] for x in rows],[x[key] for x in rows],label=key,color=color,lw=1.7)
        ax.set(title=title,xlabel='Optimizer update');ax.grid(alpha=.15)
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=9,frameon=False)
        if index in [0,1,2] and source:ax.set_yscale('log')
    import shutil
    last=train[-1]['step'] if train else 0
    save(fig,run/f'figures/training-curves-step-{last:07d}')
    for suffix in ['.png','.svg']:
        shutil.copy2(run/f'figures/training-curves-step-{last:07d}{suffix}',run/f'figures/training-curves{suffix}')
    if gen:
        fig,axes=plt.subplots(1,4,figsize=(18,4))
        for ax,key in zip(axes,['bond_valid','acf_mae','velocity_sliced_w1','motion_ratio']):
            rows=[x for x in gen if key in x]
            ax.plot([x['step'] for x in rows],[x[key] for x in rows],color=BLUE,marker='o',ms=3)
            ax.set(title=key,xlabel='Optimizer update');ax.grid(alpha=.15)
        save(fig,run/'figures/generation-curves')
    report(run)

def validation_figures(output,arrays,prediction=None,target=None,scatter=None):
    output=Path(output)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    maximum=max(float(arrays['reference_phi_psi'].max()),float(arrays['generated_phi_psi'].max()))
    for ax,key,title in [(axes[0,0],'reference_phi_psi','Reference phi/psi'),(axes[0,1],'generated_phi_psi','Generated phi/psi')]:
        h=ax.imshow(arrays[key].T,origin='lower',extent=[-180,180,-180,180],cmap='magma',aspect='auto',vmin=0,vmax=maximum)
        ax.set(title=title,xlabel='phi (degrees)',ylabel='psi (degrees)');fig.colorbar(h,ax=ax)
    difference=arrays['generated_contacts']-arrays['reference_contacts']
    h=axes[0,2].imshow(difference,cmap='RdBu_r',vmin=-1,vmax=1);fig.colorbar(h,ax=axes[0,2]);axes[0,2].set(title='CA contact probability difference',xlabel='Residue',ylabel='Residue')
    h=axes[1,0].imshow(arrays['ca_error_by_time'].T,aspect='auto',origin='lower',cmap='viridis');fig.colorbar(h,ax=axes[1,0]);axes[1,0].set(title='Stochastic sample CA discrepancy (A)',xlabel='Frame',ylabel='Residue')
    h=axes[1,1].imshow(arrays['bond_valid_by_time'].T,aspect='auto',vmin=0,vmax=1,cmap='viridis');fig.colorbar(h,ax=axes[1,1]);axes[1,1].set(title='Peptide bond validity',xlabel='Frame',ylabel='Bond')
    for key,color,label in [('reference_acf',RED,'Reference'),('generated_acf',BLUE,'Generated')]:
        a=arrays[key];axes[1,2].plot(a[:,0],a[:,1],marker='o',label=label,color=color)
    axes[1,2].set(title='Torsion temporal correlation',xlabel='Lag (frames)',ylabel='Mean cosine');axes[1,2].legend(frameon=False)
    axes[0,2].set_xticks(range(4),labels=range(1,5));axes[0,2].set_yticks(range(4),labels=range(1,5))
    axes[1,0].set_yticks(range(4),labels=range(1,5));axes[1,1].set_yticks(range(3),labels=range(1,4))
    save(fig,output/'heatmaps-and-dynamics')
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for key,color,label in [('reference_psd',RED,'Reference'),('generated_psd',BLUE,'Generated')]:
        p=arrays[key];axes[0].plot(np.fft.rfftfreq((len(p)-1)*2),p,label=label,color=color)
    axes[0].set(title='Backbone velocity spectrum',xlabel='Cycles/frame',ylabel='Normalized power');axes[0].set_yscale('log');axes[0].legend(frameon=False)
    ref=arrays['reference'][0,:,1,1,0];pred=arrays['generated'][0,:,1,1,0]
    axes[1].plot(np.arange(len(ref))*10,ref,color=RED,label='Reference')
    axes[1].plot(np.arange(len(pred))*10,pred,color=BLUE,label='One free sample')
    axes[1].set(title='Residue2 CA x coordinate',xlabel='Time (ps)',ylabel='Angstrom');axes[1].legend(frameon=False)
    if scatter:
        x,y=scatter;x=np.asarray(x);y=np.asarray(y)
        axes[2].hexbin(x,y,gridsize=45,mincnt=1,cmap='Blues');bounds=[min(x.min(),y.min()),max(x.max(),y.max())]
        axes[2].plot(bounds,bounds,'--',color=RED,lw=1);axes[2].set(title='Velocity-field fit (paired targets)',xlabel='Target velocity',ylabel='Predicted velocity')
    else:axes[2].axis('off')
    save(fig,output/'spectrum-trajectories-fit')
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for key,color,label in [('reference_slow',RED,'Reference coarse state'),('generated_slow',BLUE,'Free coarse sample')]:
        value=arrays[key][0,:,1,4]
        axes[0].plot((np.arange(len(value))*8+4.5)*10,value,label=label,color=color)
    axes[0].set(title='Slow state: residue2 translation x',xlabel='Time (ps)',ylabel='Normalized feature');axes[0].legend(frameon=False,fontsize=9)
    r=arrays['reference_residual'][... ,4:7].ravel();g=arrays['generated_residual'][...,4:7].ravel()
    bins=np.linspace(min(r.min(),g.min()),max(r.max(),g.max()),100)
    axes[1].hist(r,bins=bins,density=True,histtype='step',color=RED,label='Reference')
    axes[1].hist(g,bins=bins,density=True,histtype='step',color=BLUE,label='Generated')
    axes[1].set(title='Fast residual distribution',xlabel='Normalized translation residual',ylabel='Density',yscale='log');axes[1].legend(frameon=False)
    probabilities=np.linspace(.001,.999,501);rq=np.quantile(r,probabilities);gq=np.quantile(g,probabilities)
    axes[2].plot(rq,gq,color=BLUE);lo=min(rq.min(),gq.min());hi=max(rq.max(),gq.max())
    axes[2].plot([lo,hi],[lo,hi],'--',color=RED);axes[2].set(title='Residual distribution quantile fit',xlabel='Reference quantile',ylabel='Generated quantile')
    save(fig,output/'slow-fast-decomposition')
    if prediction is not None:
        error=np.mean((prediction-target)**2,axis=(0,1))
        fig,ax=plt.subplots(figsize=(12,3));h=ax.imshow(error,aspect='auto',cmap='viridis');fig.colorbar(h,ax=ax)
        ax.set(title='Endpoint fit MSE by residue and feature',xlabel='21 geometric features',ylabel='Residue')
        save(fig,output/'feature-fit-heatmap')

def trajectory_video(output,arrays):
    import imageio.v2 as imageio
    output=Path(output)
    data=[arrays['reference'][0,:,:,:4].reshape(-1,16,3),arrays['generated'][0,:,:,:4].reshape(-1,16,3)]
    joined=np.concatenate(data);center=joined.reshape(-1,3).mean(0);radius=max(5,float(np.max(np.abs(joined-center)))+1)
    edges=[]
    for r in range(4):edges.extend([(4*r,4*r+1),(4*r+1,4*r+2),(4*r+2,4*r+3)])
    edges.extend([(4*r+2,4*(r+1)) for r in range(3)])
    fig=plt.figure(figsize=(10.24,4.16),dpi=100)
    writer=imageio.get_writer(output/'trajectory.mp4',fps=16,codec='libx264',quality=7)
    try:
        stride=max(1,len(data[0])//128)
        for frame in range(0,len(data[0]),stride):
            fig.clear()
            for panel,(values,title) in enumerate(zip(data,['MD reference','Free model sample']),1):
                ax=fig.add_subplot(1,2,panel,projection='3d');v=values[frame]
                for a,b in edges:ax.plot(*v[[a,b]].T,color='#767676',lw=1.2)
                ax.scatter(*v.T,c=['#3775BA','#767676','#767676','#B64342']*4,s=24)
                ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
                ax.view_init(20,45);ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.set_title(f'{title}: {frame*10} ps')
            fig.tight_layout();fig.canvas.draw();pixels=np.asarray(fig.canvas.buffer_rgba())[:,:,:3];writer.append_data(pixels)
            if frame==0:imageio.imwrite(output/'trajectory-preview.png',pixels)
    finally:writer.close();plt.close(fig)

def display(value):
    return format(value,'.6g') if isinstance(value,(int,float)) else html.escape(str(value))

def report(run):
    run=Path(run);items=[]
    for path in sorted(run.glob('validation/step-*'))+([run/'final-validation'] if (run/'final-validation').exists() else []):
        if not (path/'metrics.json').exists():continue
        row=json.loads((path/'metrics.json').read_text());rel=path.relative_to(run)
        table=''.join(f'<tr><td>{html.escape(k)}</td><td>{display(v)}</td></tr>' for k,v in row['metrics'].items())
        media=''.join(f'<a href="{rel}/{p.name}"><img src="{rel}/{p.name}" loading="lazy"></a>' for p in path.glob('*.png'))
        video=f'<video controls preload="none" src="{rel}/trajectory.mp4"></video>' if (path/'trajectory.mp4').exists() else ''
        title='Final selected checkpoint' if path.name=='final-validation' else 'Checkpoint'
        items.append(f'<section><h2>{title}: update{row["step"]}</h2><table>{table}</table>{media}{video}<p><a href="{rel}/trajectories.npz">Raw trajectories and heatmap arrays</a></p></section>')
    text='''<!doctype html><meta charset="utf-8"><title>Molecular trajectory training</title>
<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:20px;color:#172b42}img{max-width:100%;border:1px solid #ddd;margin:8px 0}video{width:100%}td{padding:4px 14px;border-bottom:1px solid #eee}section{margin-top:36px}a{color:#0f4d92}</style>
    <h1>Molecular trajectory training</h1>
<p>Known initial-state-conditioned stochastic trajectory generation. Architecture is recorded with each checkpoint. Validation only. A generated trajectory is a stochastic sample, not a paired deterministic prediction.</p>
<p><a href="train.csv">Training CSV</a> · <a href="validation.csv">Validation CSV</a> · <a href="generation.csv">Generation CSV</a></p>
<img src="figures/training-curves.png"><img src="figures/generation-curves.png">'''+''.join(items)
    tmp=run/'report.tmp';tmp.write_text(text);tmp.replace(run/'report.html')
