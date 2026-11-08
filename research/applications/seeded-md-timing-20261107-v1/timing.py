"""Equal elapsed-time MD coverage, with actual online candidate generation."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from pilot import (ARMS, EQUIL_STEPS, REPLICAS, SAMPLES, SAMPLE_STRIDE, STEP_PS, choose_candidates,
                   count_bins, valid_geometry, write_json)
from geometry import setup_simulation


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PANELS = ("seeded-md-pilot-20261107-v1", "seeded-md-replication-20261107-v1")
RAPID = BASE/"rapid-evidence-20261107-v1"
THREADS = 2


def load_plan():
    rows = []
    for panel in PANELS:
        folder = BASE/panel/"inputs"
        rows.extend({**x, "folder": folder} for x in json.loads((folder/"protocol.json").read_text())["molecules"])
    meta = json.loads((RAPID/"reports/checkpoint-audit32/repaired45k/metrics.json").read_text())
    starts = [x for b in meta["sample_info"] for x in zip(b["names"], b["starts"])]
    assert [r["name"] for r in rows] == [n for n, _ in starts] and len(rows) == 32
    return [dict(row, start=starts[i][1], noise_seed=841100+i) for i,row in enumerate(rows)]


def initial_batch(torch, name, start):
    from common_v2 import DATA
    with np.load(DATA/f"{name}_cache.npz") as d:
        batch = {k: torch.as_tensor(d[k][start:start+1]).unsqueeze(0) for k in ("trans", "rots", "torsions")}
        for k in ("seqres", "torsion_mask"):
            batch[k] = torch.as_tensor(d[k]).unsqueeze(0)
    batch["mask"] = torch.ones((1, 4))
    return batch


def model_candidates(model, batch, original, mask, noise_seed):
    import torch
    from common_v2 import decode_latents, true_atom14
    from dynamics import tensors
    from model import generate

    with torch.no_grad():
        prepared, z, geom, feature_mask = tensors(batch, model)
        np.testing.assert_allclose(true_atom14(batch, prepared).numpy()[0, 0], original, rtol=0, atol=5e-5)
        values, _, _ = generate(model, z[:, 0], geom, batch["seqres"], feature_mask, 256, noise_seed, 16, 32)
        raw = values*model.scale + model.mean
        raw[:, 0] = prepared["latents"][:, 0]
        xyz = decode_latents(raw, prepared, batch["seqres"])[0].numpy()[0].astype(float)
    generated_at = time.monotonic()
    if not valid_geometry(xyz[1:], mask).any():
        return np.repeat(original[None], REPLICAS, axis=0), [0]*REPLICAS, 0, generated_at
    chosen, eligible = choose_candidates(xyz, original, mask)
    return xyz[chosen], chosen, eligible, generated_at


def pipeline(arm, row, model, batch, original, mask):
    begin = time.monotonic()
    if arm == "model":
        starts, chosen, eligible, generated_at = model_candidates(model, batch, original, mask, row["noise_seed"])
        generation_seconds = generated_at-begin
        screening_seconds = time.monotonic()-generated_at
    else:
        starts = np.repeat(original[None], REPLICAS, axis=0)
        chosen, eligible = [0]*REPLICAS, None
        generation_seconds = screening_seconds = 0.
    ready_at = time.monotonic()-begin
    contexts, topology, equil, equil_t = [], None, [], []
    for coordinates, seed in zip(starts, row["seeds"]):
        sim, snapshot, current = setup_simulation(row["folder"], row["name"], coordinates, mask, seed, THREADS)
        assert topology is None or topology == current
        topology = current
        sim.step(EQUIL_STEPS)
        equil.append(snapshot())
        equil_t.append(time.monotonic()-begin)
        contexts.append((sim, snapshot))
    positions, times, replicas = [], [], []
    for _ in range(SAMPLES):
        for replica, (sim, snapshot) in enumerate(contexts):
            sim.step(SAMPLE_STRIDE)
            positions.append(snapshot())
            times.append(time.monotonic()-begin)
            replicas.append(replica)
    assert all(sim.currentStep == EQUIL_STEPS+SAMPLE_STRIDE*SAMPLES for sim, _ in contexts)
    xyz = np.stack(positions)
    assert np.isfinite(xyz).all()
    return {
        "trajectory": xyz, "times": np.array(times), "replicas": np.array(replicas),
        "starts": starts, "start_available": np.repeat(ready_at, REPLICAS),
        "equilibrated": np.stack(equil), "equil_available": np.array(equil_t),
        "full_seconds": times[-1], "generation_seconds": generation_seconds,
        "screening_seconds": screening_seconds, "eligible": eligible, "frames_selected": chosen, "topology": topology,
    }


def score(data, mask, elapsed):
    starts = []
    for arm in ARMS:
        x = data[arm]
        starts.extend(x["starts"][x["start_available"] <= elapsed])
        starts.extend(x["equilibrated"][x["equil_available"] <= elapsed])
    production = {}
    for arm in ARMS:
        x = data[arm]
        keep = x["times"] <= elapsed
        selected = x["trajectory"][keep]
        production[arm] = (selected[valid_geometry(selected, mask)], int(keep.sum()))
    result = {"budget_seconds": elapsed, "arms": {}}
    for arm, (valid, n) in production.items():
        record = {"observations": n, "production_ps": n*SAMPLE_STRIDE*STEP_PS,
                  "valid_observations": len(valid), "bins": {}}
        for width in (60, 90):
            initial = set(count_bins(np.stack(starts), width)) if starts else set()
            visits = {bin_id for bin_id, count in Counter(count_bins(valid, width)).items() if count >= 3}
            record["bins"][str(width)] = len(visits-initial)
        result["arms"][arm] = record
    return result


def worker(rank, world):
    import torch
    import openmm.app  # Load the same engine libraries before either timer starts.
    from model import HierarchicalDynamics

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    began = time.monotonic()
    state = torch.load(RAPID/"artifacts/checkpoint-step-0045000.pt", map_location="cpu", weights_only=False, mmap=True)
    assert state["step"] == 45000 and state["ema"]
    model = HierarchicalDynamics(checkpointing=False).eval()
    model.load_state_dict({**state["model"], **state["ema"]})
    del state
    load_time = time.monotonic()-began
    output = ROOT/"runs"
    output.mkdir(exist_ok=True)
    for i, row in enumerate(load_plan()):
        if i % world != rank:
            continue
        name = row["name"]
        path = output/f"{name}.json"
        if path.exists():
            continue
        with np.load(row["folder"]/f"{name}.npz") as x:
            original, mask = x["original"], x["mask"]
        batch = initial_batch(torch, name, row["start"])
        order = ARMS if i % 2 == 0 else ARMS[::-1]
        arms = {}
        for arm in order:
            arms[arm] = pipeline(arm, row, model, batch, original, mask)
            print(json.dumps({"molecule": name, "arm": arm,
                              "total_seconds": arms[arm]["full_seconds"],
                              "generation_seconds": arms[arm]["generation_seconds"]}), flush=True)
        assert arms["original"]["topology"] == arms["model"]["topology"]
        elapsed = min(x["full_seconds"] for x in arms.values())
        report = {"molecule": name, "order": list(order), "load_seconds": load_time,
                  "timing": {arm: {key: a[key] for key in ("full_seconds", "generation_seconds",
                              "screening_seconds", "eligible", "frames_selected")} for arm, a in arms.items()},
                  "equal_time": score(arms, mask, elapsed)}
        arrays = {f"{arm}_{key}": a[key] for arm, a in arms.items() for key in (
            "trajectory", "times", "replicas", "starts", "start_available", "equilibrated", "equil_available")}
        np.savez_compressed(output/f"{name}.npz", **arrays, mask=mask)
        write_json(path, report)


def summarize(rows):
    draws = np.random.default_rng(20261107).integers(len(rows), size=(10000, len(rows)))
    summary = {}
    for width in ("60", "90"):
        values = {arm: np.array([r["equal_time"]["arms"][arm]["bins"][width] for r in rows]) for arm in ARMS}
        delta = values["model"] - values["original"]
        summary[width] = {"mean": {arm: float(x.mean()) for arm, x in values.items()},
                          "delta": float(delta.mean()), "ci95": np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                          "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}
    return summary


def aggregate():
    plan = load_plan()
    rows = [json.loads((ROOT/"runs"/f"{r['name']}.json").read_text()) for r in plan]
    assert len({r["molecule"] for r in rows}) == 32
    for r in rows:
        assert r["equal_time"]["budget_seconds"] == min(a["full_seconds"] for a in r["timing"].values())
    result = {"checkpoint_step": 45000, "cpu_threads_per_workflow": THREADS, "molecules": 32,
              "budget_rule": "per molecule, shorter measured time to complete three 50 ps replicates",
              "clock": "monotonic wall time including generation, screening, minimization, 5 ps relaxation, integration, snapshots; model preloaded",
              "summary": summarize(rows), "first_8": summarize(rows[:8]), "remaining_24": summarize(rows[8:]),
              "median_budget_seconds": float(np.median([r["equal_time"]["budget_seconds"] for r in rows])),
              "median_model_generation_seconds": float(np.median([r["timing"]["model"]["generation_seconds"] for r in rows])),
              "median_model_screening_seconds": float(np.median([r["timing"]["model"]["screening_seconds"] for r in rows])),
              "mean_production_ps_per_molecule": {a: float(np.mean([r["equal_time"]["arms"][a]["production_ps"] for r in rows])) for a in ARMS},
              "no_production_molecules": {a: sum(r["equal_time"]["arms"][a]["observations"] == 0 for r in rows) for a in ARMS},
              "per_molecule": rows}
    write_json(ROOT/"results.json", result)
    print(json.dumps({k: v for k,v in result.items() if k != "per_molecule"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    else:
        worker(int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]))
