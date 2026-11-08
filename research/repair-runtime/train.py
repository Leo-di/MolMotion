import argparse
from contextlib import contextmanager,nullcontext
import csv
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common_v2 import ROOT,DATA,Trajectories,verify_inputs
from dynamics import tensors,fit_normalization,loss_terms
from model import make_model
from evaluation import free_validation
from figures import curves,validation_figures,trajectory_video,report

STOP=False
def stop_handler(signum,frame):
    global STOP;STOP=True

def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)

def atomic_save(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)

def log(run,name,row):
    row={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),**row}
    with (run/f'{name}.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    path=run/f'{name}.csv'
    if path.exists():
        with path.open() as f:keys=next(csv.reader(f))
    else:keys=list(row)
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore')
        if path.stat().st_size==0:writer.writeheader()
        writer.writerow(row)
    print(json.dumps({name:row}),flush=True)

@contextmanager
def ema_weights(model,shadow):
    if shadow is None:yield;return
    original={n:p.data for n,p in model.named_parameters()}
    try:
        for n,p in model.named_parameters():p.data=shadow[n]
        yield
    finally:
        for n,p in model.named_parameters():p.data=original[n]

@torch.no_grad()
def validate(model,dataset,future,step):
    model.eval();rows=[];last=None
    batches=max(2,math.ceil(len(dataset.names)/4))
    for i in range(batches):
        batch,record=dataset.batch(4,future+1,81000+i,balanced=True,name_offset=i*4)
        torch.manual_seed(71000+i)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,numbers,regression,aux=loss_terms(model,batch,step,True)
        rows.append({k:float(v) for k,v in {**numbers,**regression}.items()});last=aux
        last['sample_info']=record
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]},last

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True);p.add_argument('--phase',required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--future',type=int,default=256)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--width',type=int,default=768);p.add_argument('--slow-layers',type=int,default=12)
    p.add_argument('--fast-layers',type=int,default=16);p.add_argument('--heads',type=int,default=12)
    p.add_argument('--architecture',choices=['hierarchical','single-scale'],default='hierarchical');p.add_argument('--joint-ode',type=int,default=64)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--warmup',type=int,default=2000);p.add_argument('--schedule-steps',type=int,default=250000)
    p.add_argument('--validate-every',type=int,default=1000);p.add_argument('--generate-every',type=int,default=5000)
    p.add_argument('--checkpoint-every',type=int,default=5000);p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--generation-clips',type=int,default=8)
    p.add_argument('--slow-ode',type=int,default=16);p.add_argument('--fast-ode',type=int,default=32)
    p.add_argument('--warm-restart');p.add_argument('--restart-warmup',type=int,default=500);p.add_argument('--restart-origin-step',type=int,default=0);p.add_argument('--max-grad-norm-raw',type=float,default=100.);
    p.add_argument('--max-update-retries',type=int,default=3);p.add_argument('--max-dropped-updates',type=int,default=20);
    p.add_argument('--resume');p.add_argument('--initialize');p.add_argument('--allow-data-transition',action='store_true')
    p.add_argument('--memory-gib',type=float,default=60);p.add_argument('--max-hours',type=float,default=60)
    p.add_argument('--normalization',default='normalization.pt');p.add_argument('--ema',action='store_true')
    p.add_argument('--minimum-steps',type=int,default=0);p.add_argument('--patience',type=int,default=0)
    p.add_argument('--final-clips',type=int,default=0)
    p.add_argument("--repair",choices=["none","slow_dropout","geometry_detach_slow"],default="none")
    return p.parse_args()

def main():
    args=parse();rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    torch.set_num_threads(4)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2))
    _,total_memory=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(min(args.memory_gib*2**30/total_memory,.95))
    torch.backends.cuda.matmul.allow_tf32=True
    signal.signal(signal.SIGUSR1,stop_handler);signal.signal(signal.SIGTERM,stop_handler)
    run=ROOT/'runs'/args.run
    if rank==0:run.mkdir(parents=True,exist_ok=bool(args.resume))
    if world>1:dist.barrier()
    fingerprints=verify_inputs() if rank==0 else None
    train=Trajectories('train','cuda');val=Trajectories('val','cuda')
    assert not set(train.names)&set(val.names)
    norm_path=ROOT/args.normalization
    if rank==0 and not norm_path.exists():
        cpu_train=Trajectories('train','cpu')
        norm=fit_normalization(cpu_train);del cpu_train
        atomic_save(norm_path,norm)
    if world>1:dist.barrier()
    norm=torch.load(norm_path,map_location='cpu',weights_only=False)
    assert set(norm['training_names'])<=set(train.names)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    model=make_model(args.architecture,args.width,args.slow_layers,args.fast_layers,args.heads).cuda();model.joint_ode=args.joint_ode
    for key in ['mean','scale','residual_scale']:getattr(model,key).copy_(norm[key].cuda())
    model.repair=args.repair
    if args.repair=='slow_dropout':
        for block in model.slow.blocks:
            block.time.dropout_p=.1;block.space.dropout_p=.1
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.01,fused=True)
    start=0;best=float('inf');best_generation=float('inf');bad=0;shadow=None;lr_scale=1.;dropped_updates=0
    if args.initialize:
        state=torch.load(args.initialize,map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('ema') or state['model']);del state
    if args.warm_restart:
        state=torch.load(args.warm_restart,map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('ema') or state['model']);start=state['step'];del state
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        for k in ['width','slow_layers','fast_layers','heads','architecture']:
            assert state['config'].get(k,'hierarchical' if k=='architecture' else None)==vars(args)[k],k
        if args.allow_data_transition:
            assert set(state['train_names'])<=set(train.names)
        else:
            assert state['train_names']==train.names and state['validation_names']==val.names
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['step'];lr_scale=state.get('lr_scale',1.);dropped_updates=state.get('dropped_updates',0)
        if state['config']['phase']==args.phase:
            best=state['best'];best_generation=state.get('best_generation',float('inf'));bad=state.get('bad',0)
        if args.ema and rank==0:
            shadow={n:v.cuda() for n,v in (state.get('ema') or state['model']).items() if n in dict(model.named_parameters())}
        del state
    if args.ema and rank==0 and shadow is None:shadow={n:p.detach().clone() for n,p in model.named_parameters()}
    parameters=sum(p.numel() for p in model.parameters())
    cfg={**vars(args),'parameters':parameters,'world_size':world,'global_batch':args.batch*args.accumulate*world,
         'train_names':train.names,'validation_names':val.names,'data':str(DATA),'normalization_training_names':norm['training_names'],
         'test_loaded':False,'generation':'joint full-window flow conditioned only on initial state' if args.architecture=='single-scale' else 'causal coarse flow then residual flow',
         'future_target_condition_during_training':'single-scale network sees noisy joint latent and known initial condition only' if args.architecture=='single-scale' else 'true coarse during training; generated coarse only at inference'}
    if rank==0:
        atomic_json(run/f'config-from-step-{start}.json',cfg)
        print(json.dumps({'parameters':parameters,'start':start,'global_batch':cfg['global_batch'],'phase':args.phase}),flush=True)
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
    begun=time.monotonic();last_log=begun;initial_step=start;stop_reason=None

    def checkpoint_state(step):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),ema=shadow,step=step,
            best=best,best_generation=best_generation,bad=bad,lr_scale=lr_scale,dropped_updates=dropped_updates,config=vars(args),train_names=train.names,
            validation_names=val.names,normalization=norm,data_fingerprints=fingerprints,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())

    step=start;update_retries=0
    while step<args.steps:
        step+=1
        model.train();optimizer.zero_grad(set_to_none=True);rows=[]
        warm=min(1.,(step-args.restart_origin_step)/max(args.restart_warmup,1)) if args.restart_origin_step else min(1.,step/max(args.warmup,1));fraction=min(1.,max(0.,(step-args.warmup)/max(1,args.schedule_steps-args.warmup)))
        lr=args.lr*warm*(.1+.9*.5*(1+math.cos(math.pi*fraction)))*lr_scale
        for group in optimizer.param_groups:group['lr']=lr
        for micro in range(args.accumulate):
            seed=args.seed*10000000+step*world*args.accumulate+rank*args.accumulate+micro+update_retries*1000000000000
            batch,_=train.batch(args.batch,args.future+1,seed)
            torch.manual_seed(seed)
            sync=nullcontext() if world==1 or micro==args.accumulate-1 else wrapped.no_sync()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):total,numbers,regression,_=loss_terms(wrapped,batch,step)
                assert torch.isfinite(total),f'nonfinite loss at{step}'
                (total/args.accumulate).backward()
            rows.append({k:float(v.detach()) for k,v in {**numbers,**regression}.items()})
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
        if not torch.isfinite(grad) or float(grad)>args.max_grad_norm_raw:
            dropped_updates+=1;update_retries+=1;lr_scale=max(lr_scale*.5,.0625)
            fatal=update_retries>args.max_update_retries or dropped_updates>args.max_dropped_updates
            if rank==0:
                incident=dict(attempted_update=step,gradient_norm=float(grad),threshold=args.max_grad_norm_raw,optimizer_step_applied=False,retry=update_retries,dropped_updates=dropped_updates,new_lr_scale=lr_scale,fatal=fatal)
                log(run,'anomalies',incident)
                atomic_save(run/'emergency-before-gradient-spike.pt',checkpoint_state(step-1))
                atomic_json(run/'gradient-spike.json',incident)
            optimizer.zero_grad(set_to_none=True)
            if world>1:dist.barrier()
            if fatal:raise FloatingPointError('Bounded gradient recovery budget exhausted; no anomalous optimizer update applied')
            step-=1
            continue
        optimizer.step();update_retries=0
        if shadow is not None and step%10==0:
            for n,p in model.named_parameters():shadow[n].lerp_(p.detach(),1-.999**10)
        if step==start+1 or step%args.log_every==0:
            numbers={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            if world>1:
                packed=torch.tensor(list(numbers.values()),device='cuda')
                dist.all_reduce(packed);packed/=world
                numbers=dict(zip(numbers,packed.cpu().tolist()))
            memory=torch.tensor([torch.cuda.max_memory_allocated()/2**30,torch.cuda.memory_reserved()/2**30],device='cuda')
            if world>1:dist.all_reduce(memory,op=dist.ReduceOp.MAX)
            if rank==0:
                elapsed=time.monotonic()-begun
                numbers.update(step=step,phase=args.phase,lr=lr,grad_norm=float(grad),
                    updates_per_second=(step-start)/max(elapsed,1e-8),clips_seen=(step-initial_step)*cfg['global_batch'],
                    frames_seen=(step-initial_step)*cfg['global_batch']*args.future,
                    dropped_updates=dropped_updates,lr_scale=lr_scale,gpu_peak_gib=float(memory[0]),gpu_reserved_gib=float(memory[1]),wall_hours=elapsed/3600)
                log(run,'train',numbers)
        do_val=step%args.validate_every==0 or step==args.steps
        do_generate=step%args.generate_every==0 or step==args.steps
        do_save=step%args.checkpoint_every==0 or step==args.steps
        if do_val or do_generate or do_save:
            if world>1:dist.barrier()
            if rank==0:
                with ema_weights(model,shadow):
                    if do_val:
                        scores,aux=validate(model,val,args.future,step)
                        log(run,'validation',dict(step=step,phase=args.phase,**scores))
                        if scores['total']<best:
                            best=scores['total'];bad=0
                            atomic_save(run/'best.pt',dict(model=model.state_dict(),step=step,config=vars(args),normalization=norm))
                        else:bad+=1
                    if do_generate:
                        directory=run/'validation'/f'step-{step:07d}'
                        scores,arrays=free_validation(model,val,args.future,directory,step,args.generation_clips,args.slow_ode,args.fast_ode)
                        log(run,'generation',dict(step=step,phase=args.phase,**scores))
                        _,aux=validate(model,val,args.future,step)
                        np.savez_compressed(directory/'fitting-arrays.npz',prediction=aux['prediction'].cpu().numpy(),
                            target=aux['target'].cpu().numpy(),velocity_target=aux['scatter_target'].cpu().numpy(),
                            velocity_prediction=aux['scatter_prediction'].cpu().numpy())
                        atomic_json(directory/'fitting-sample-info.json',aux['sample_info'])
                        validation_figures(directory,arrays,aux['prediction'].cpu().numpy(),aux['target'].cpu().numpy(),
                            (aux['scatter_target'].cpu().numpy(),aux['scatter_prediction'].cpu().numpy()))
                        trajectory_video(directory,arrays)
                        score=scores['acf_mae']+abs(math.log(max(scores['motion_ratio'],.01)))+2*(1-scores['bond_valid'])+scores['nonadjacent_clash_frame_fraction']+scores['phi_psi_js']
                        if score<best_generation:
                            best_generation=score
                            atomic_save(run/'best-generation.pt',dict(model=model.state_dict(),step=step,config=vars(args),normalization=norm,selection_score=score))
                if do_save:
                    atomic_save(run/'last.pt',checkpoint_state(step))
                    if step%5000==0:
                        import shutil
                        shutil.copyfile(run/'last.pt',run/f'checkpoint-step-{step:07d}.pt')
                        rolling=[p for p in sorted(run.glob('checkpoint-step-*.pt')) if int(p.stem.split('-')[-1])%25000]
                        for old_checkpoint in rolling[:-3]:old_checkpoint.unlink()
                curves(run)
                atomic_json(run/'status.json',dict(status='running',step=step,target=args.steps,phase=args.phase,
                    best_validation=best,best_generation=best_generation,parameters=parameters,world_size=world,updated_at=time.time()))
            if world>1:dist.barrier()
        flag=int(STOP or (time.monotonic()-begun)>args.max_hours*3600)
        if rank==0 and args.patience and step>=args.minimum_steps and bad>=args.patience:flag=2
        stop=torch.tensor(flag,device='cuda')
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():
            if rank==0:atomic_save(run/'last.pt',checkpoint_state(step))
            if stop.item()==2:stop_reason='validation_early_stop';break
            if rank==0:atomic_json(run/'status.json',dict(status='checkpointed_time_or_signal',step=step,target=args.steps))
            if world>1:dist.barrier();dist.destroy_process_group()
            raise SystemExit(75)
    if rank==0:
        atomic_save(run/'last.pt',checkpoint_state(step))
        if args.final_clips:
            state=torch.load(run/'best-generation.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(state['model']);selected_step=state['step'];del state
            directory=run/'final-validation'
            scores,arrays=free_validation(model,val,args.future,directory,selected_step,args.final_clips,args.slow_ode,args.fast_ode)
            validation_figures(directory,arrays);trajectory_video(directory,arrays)
        atomic_json(run/'status.json',dict(status='completed',step=step,target=args.steps,reason=stop_reason or 'budget_complete',parameters=parameters))
        (run/'COMPLETE').write_text(f'Completed at update{step}\n')
        curves(run)
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
