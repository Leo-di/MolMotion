"""Official MDGen model and transport on the RSI paired trajectory protocol."""
import argparse
from contextlib import nullcontext
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from common_v2 import (
    DATA, Trajectories, atom14_mask, decode_latents, prepare_batch, true_atom14, verify_inputs,
)
from evaluation import metrics
from mdgen.model.latent_model import LatentMDGenModel
from mdgen.parsing import parse_train_args
from mdgen.transport.transport import Sampler, create_transport

ROOT = Path(__file__).resolve().parent
PAIRED = Path(os.environ["PAIRED_REFERENCE"])
PUBLISHED = ROOT / "inputs/forward_sim.ckpt"
FRAMES = 257
SEED = 43
METRICS = ("bond_valid", "acf_mae", "velocity_sliced_w1", "motion_ratio",
           "nonadjacent_clash_frame_fraction", "phi_psi_js", "velocity_psd_l1")


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint(path):
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location="cpu", weights_only=True)


def upstream_config(overrides):
    argv = sys.argv
    try:
        sys.argv = [argv[0], "--train_split", str(DATA / "train.csv"),
                    "--val_split", str(DATA / "val.csv"), "--data_dir", str(DATA)]
        args = parse_train_args()
    finally:
        sys.argv = argv
    vars(args).update(overrides)
    args.num_frames = FRAMES
    args.grad_checkpointing = False
    assert args.sim_condition and args.crop == 4
    assert not any(getattr(args, key, False) for key in (
        "design", "tps_condition", "inpainting", "mpnn", "dynamic_mpnn", "no_frames",
        "cond_interval", "no_offsets", "no_torsion", "no_design_torsion",
        "supervise_no_torsions", "supervise_all_torsions",
    ))
    return args


def published_config(state):
    args = state["hyper_parameters"]["args"]
    return dict(args) if isinstance(args, dict) else vars(args).copy()


def conditioning(prepared):
    kwargs = dict(prepared["model_kwargs"])
    # Do not pass any future frame into the network.
    kwargs.pop("end_frames", None)
    return kwargs


def limit_gpu_memory():
    if os.environ.get("MDGEN_MEMORY_GIB"):
        cap = float(os.environ["MDGEN_MEMORY_GIB"]) * 2**30
        torch.cuda.set_per_process_memory_fraction(
            min(cap / torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory, .95)
        )


def model_and_transport(args, device):
    if device == "cuda":
        limit_gpu_memory()
    return (LatentMDGenModel(args, 21).to(device),
            create_transport(args, args.path_type, args.prediction, None))


def evaluate(arm, device, method, steps, limit, path=None):
    source = path or PUBLISHED
    saved = checkpoint(source)
    if arm == "published":
        config = upstream_config(published_config(saved))
        if "ema" in saved:
            weights = saved["ema"]["params"]
        else:
            weights = {k.removeprefix("model."): v for k, v in saved["state_dict"].items()
                       if k.startswith("model.")}
    else:
        config = upstream_config(saved["upstream"])
        weights = saved["ema"]
    if "time_embed" in weights:
        weights = dict(weights)
        weights["time_embed"] = weights["time_embed"][:, :FRAMES]
    model, transport = model_and_transport(config, device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    train_step = saved.get("step")
    del weights, saved
    dataset = Trajectories("val", device)
    paired = json.loads((PAIRED / "results.json").read_text())
    with np.load(PAIRED / "single-scale/trajectories.npz") as archive:
        reference, sequences, mask = [archive[k] for k in ("reference", "sequence_ids", "atom_mask")]
    with np.load(PAIRED / "hierarchical/trajectories.npz") as archive:
        assert np.array_equal(archive["reference"], reference)
    metadata = json.loads((PAIRED / "single-scale/metrics.json").read_text())["sample_info"]
    output = ROOT / "reports" / (arm if limit == 32 else f"{arm}-probe")
    output.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(transport).sample_ode(sampling_method=method, num_steps=steps)
    preds, per = [], []
    max_ref_error = 0.
    started = time.monotonic()
    for offset in range(0, limit, 2):
        batch, record = dataset.batch(2, FRAMES, 91000 + offset, balanced=True, name_offset=offset)
        assert record == metadata[offset // 2], (record, metadata[offset // 2])
        with torch.no_grad():
            prepared = prepare_batch(batch)
            gt = true_atom14(batch, prepared).cpu().numpy()
            sl = slice(offset, offset + 2)
            assert np.array_equal(batch["seqres"].cpu().numpy(), sequences[sl])
            assert np.array_equal(atom14_mask(batch["seqres"], FRAMES)[:, 0].cpu().numpy(), mask[sl])
            max_ref_error = max(max_ref_error, float(np.max(np.abs(gt - reference[sl]))))
            assert max_ref_error < 5e-5, max_ref_error
            torch.manual_seed(40100 + offset)
            noise = torch.randn_like(prepared["latents"])
            samples = sampler(noise, model.forward_inference, **conditioning(prepared))[-1]
            samples[:, 0] = prepared["latents"][:, 0]
            decoded, _ = decode_latents(samples, prepared, batch["seqres"])
            assert torch.isfinite(decoded).all()
            pred = decoded.cpu().numpy()
        preds.append(pred)
        for index, name in enumerate(record["names"]):
            k = index + offset
            score, _ = metrics(pred[index:index + 1], reference[k:k + 1], sequences[k:k + 1], mask[k:k + 1])
            per.append(dict(molecule=name, **score))
        print(json.dumps(dict(arm=arm, clips_done=offset+2, seconds=time.monotonic()-started)), flush=True)
    prediction = np.concatenate(preds)
    pooled, _ = metrics(prediction, reference[:limit], sequences[:limit], mask[:limit])
    for ours in paired["results"].values():
        assert [row["molecule"] for row in per] == [row["molecule"] for row in ours["per_molecule"][:limit]]
    if limit == 32:
        np.savez_compressed(output / "trajectories.npz", generated=prediction,
                            reference=reference, sequence_ids=sequences, atom_mask=mask)
    comparison = {}
    bootstrap = np.random.default_rng(93342).integers(limit, size=(10000, limit))
    for name, ours in paired["results"].items():
        per_molecule = {}
        for key in METRICS:
            differences = np.asarray([p[key] - q[key] for p, q in zip(per, ours["per_molecule"])])
            per_molecule[key] = dict(mean=float(differences.mean()),
                                     ci95=np.quantile(differences[bootstrap].mean(1), [.025,.975]).tolist())
        comparison[name] = dict(
            pooled={key: pooled[key] - ours["metrics"][key] for key in METRICS},
            per_molecule=per_molecule,
        )
    result = dict(
        status="completed", arm=arm, checkpoint=str(source), training_step=train_step,
        checkpoint_sha256=file_sha256(source),
        parameters=parameters, solver=method, solver_grid_points=steps, model_dtype="float32",
        device=device, conditioning="sequence and first frame only", future_frames=FRAMES-1,
        frame_interval_ps=10, development_molecules=limit, test_loaded=False,
        reference_max_coordinate_error_angstrom=max_ref_error, sample_info=metadata[:limit//2],
        training_scope=("Published original data and training budget" if arm == "published" else
                        "Fresh training, exact paired training split and clip stream"),
        pooled=pooled, per_molecule=per, mdgen_minus_30k_reference_NOT_budget_matched=comparison,
        seconds=time.monotonic()-started,
    )
    save_json(output / "results.json", result)
    print(json.dumps(dict(arm=arm, results=str(output / "results.json"), metrics=pooled)), flush=True)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train(target, batch_size, accumulation, run, device):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=45))
    assert world * batch_size * accumulation == 64 or target <= 5
    out = ROOT / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    fingerprints = verify_inputs() if rank == 0 else None
    train_data = Trajectories("train", device)
    val_names = (DATA / "val.csv").read_text().splitlines()[1:]
    assert len(train_data.names) == 256 and len(val_names) == 32
    config = upstream_config(json.loads((ROOT / "inputs/upstream-config.json").read_text()))
    torch.manual_seed(SEED)
    model, transport = model_and_transport(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9,.95), weight_decay=.01,
                                  fused=device=="cuda")
    shadow = {key: p.detach().clone() for key, p in model.named_parameters()} if rank == 0 else {}
    last = out / "last.pt"
    start = 0
    if last.exists():
        saved = checkpoint(last)
        assert (saved["world"], saved["batch_size"], saved["accumulation"]) == (world,batch_size,accumulation)
        assert saved["train_names"] == train_data.names
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if rank == 0:
            for name in shadow:
                shadow[name].copy_(saved["ema"][name])
        start = saved["step"]
        del saved
    if rank == 0:
        save_json(out / "protocol.json", dict(
            target_updates=target, parameters=sum(p.numel() for p in model.parameters()),
            world=world, batch_size=batch_size, accumulation=accumulation,
            global_batch=world*batch_size*accumulation, seed=SEED, future=FRAMES-1,
            train_names=train_data.names, validation_names=[v.split(",")[0] for v in val_names],
            data_fingerprints=fingerprints, test_loaded=False, upstream_config=vars(config),
            optimizer="AdamW, beta=(.9,.95), lr=5e-5, warmup=2000, cosine horizon=100000",
        ))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                      gradient_as_bucket_view=True) if world > 1 else model
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    begun = time.monotonic()
    model.train()
    for step in range(start+1, target+1):
        optimizer.zero_grad(set_to_none=True)
        warm = min(1., step/2000)
        fraction = max(0., (step-2000)/98000)
        lr = 5e-5 * warm * (.1 + .45 * (1 + math.cos(math.pi*fraction)))
        optimizer.param_groups[0]["lr"] = lr
        losses = []
        for micro in range(accumulation):
            seed = SEED*10000000 + step*world*accumulation + rank*accumulation + micro
            batch, _ = train_data.batch(batch_size, FRAMES, seed)
            torch.manual_seed(seed)
            prepared = prepare_batch(batch)
            sync = wrapped.no_sync() if world > 1 and micro < accumulation-1 else nullcontext()
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
            with sync, precision:
                loss = transport.training_losses(wrapped, prepared["latents"],
                                                 mask=prepared["loss_mask"],
                                                 model_kwargs=conditioning(prepared))["loss"].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {step}, rank {rank}")
                (loss/accumulation).backward()
                losses.append(loss.detach())
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if rank == 0:
            if step % 10 == 0:
                for key,p in model.named_parameters():
                    shadow[key].lerp_(p.detach(), 1-.999**10)
            if step == start+1 or step % 100 == 0 or step == target:
                values = dict(step=step, loss=float(torch.stack(losses).mean()), lr=lr,
                              raw_grad_norm=float(grad), updates_per_second=(step-start)/(time.monotonic()-begun),
                              gpu_peak_gib=(torch.cuda.max_memory_allocated()/2**30 if device=="cuda" else None))
                with (out / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(values) + "\n")
                print(json.dumps(values), flush=True)
        if world > 1:
            stopping = torch.tensor(int(stop), device=device)
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            stop = bool(stopping.item())
        if step % 500 == 0 or step == target or stop:
            if rank == 0:
                ema = {**model.state_dict(), **shadow}
                temporary = last.with_suffix(".tmp")
                torch.save(dict(model=model.state_dict(), ema=ema, optimizer=optimizer.state_dict(),
                                step=step, upstream=vars(config), train_names=train_data.names,
                                fingerprints=fingerprints, world=world, batch_size=batch_size,
                                accumulation=accumulation), temporary)
                temporary.replace(last)
            if world > 1:
                dist.barrier()
        if stop:
            break
    if world > 1:
        dist.destroy_process_group()
    return 75 if stop and step < target else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "published", "matched", "train"))
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--method", choices=("dopri5","euler"), default="euler")
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--limit", type=int, choices=(2,32), default=32)
    parser.add_argument("--target", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--run", default="matched-seed43")
    opts = parser.parse_args()
    torch.set_num_threads(opts.threads)
    if opts.mode == "inspect":
        state = checkpoint(PUBLISHED)
        config = published_config(state)
        keys = ("embed_dim","num_layers","mha_heads","sim_condition","prepend_ipa","abs_pos_emb",
                "abs_time_emb","num_frames","crop","path_type","prediction","sampling_method","ema")
        # Only the scientific flags are carried into the fresh run.
        resolved = upstream_config(config)
        save_json(ROOT / "inputs/upstream-config.json", {key: getattr(resolved,key) for key in keys})
        save_json(ROOT / "inputs/published-metadata.json", dict(
            sha256=file_sha256(PUBLISHED), original_frames=config["num_frames"],
            original_global_step=state.get("global_step"), checkpoint_ema="ema" in state,
            flags={key: getattr(resolved,key) for key in keys},
        ))
    elif opts.mode == "train":
        return train(opts.target, opts.batch, opts.accumulation, opts.run, opts.device)
    else:
        source = ROOT / "runs" / opts.run / "last.pt" if opts.mode == "matched" else None
        evaluate(opts.mode, opts.device, opts.method, opts.steps, opts.limit, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
