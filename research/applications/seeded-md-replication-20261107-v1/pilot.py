"""Paired CPU OpenMM pilot of model-selected versus original starting conformations."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import random

import numpy as np


ROOT = Path(__file__).resolve().parent
REPLICAS = 3
OFFSET = 8
MOLECULES = 24
STEP_PS = .002
EQUIL_STEPS = 2500
SAMPLE_STRIDE = 250
SAMPLES = 100
ARMS = ("original", "model")
CONSTRAINT_WINDOW = (1.1, 1.6)


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def valid_geometry(x, mask):
    distance = np.linalg.norm(x[:, :-1, 2] - x[:, 1:, 0], axis=-1)
    lo, hi = CONSTRAINT_WINDOW
    ok = ((distance >= lo) & (distance <= hi)).all(axis=1)
    for a, b in ((0, 2), (0, 3), (1, 3)):
        pair = np.linalg.norm(x[:, a, :, None] - x[:, b, None, :], axis=-1)
        present = mask[a, :, None] & mask[b, None, :]
        ok &= ~((pair < 1.2) & present).any(axis=(1, 2))
    return ok


def dihedral(a, b, c, d):
    axis = c-b
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(1e-8)
    v, w = a-b, d-c
    v -= (v*axis).sum(axis=-1, keepdims=True)*axis
    w -= (w*axis).sum(axis=-1, keepdims=True)*axis
    return np.arctan2((np.cross(axis, v)*w).sum(axis=-1), (v*w).sum(axis=-1))


def torsions(x):
    angles = []
    for i in (1, 2):
        angles.append(dihedral(x[:, i-1, 2], x[:, i, 0], x[:, i, 1], x[:, i, 2]))
        angles.append(dihedral(x[:, i, 0], x[:, i, 1], x[:, i, 2], x[:, i+1, 0]))
    return np.stack(angles, axis=-1)


def choose_candidates(trajectory, original, mask):
    eligible = np.flatnonzero(valid_geometry(trajectory[1:], mask)) + 1
    if not len(eligible):
        raise ValueError("No admissible model frames")
    theta = torsions(trajectory[eligible])
    fixed = torsions(original[None])[0]
    separation = lambda angle: (1-np.cos(theta-angle)).mean(axis=1)
    nearest = separation(fixed)
    selected = []
    for _ in range(min(REPLICAS, len(eligible))):
        pos = int(nearest.argmax())
        selected.append(int(eligible[pos]))
        nearest = np.minimum(nearest, separation(theta[pos]))
        nearest[pos] = -np.inf
    if len(selected) < REPLICAS:
        selected.extend([selected[0]]*(REPLICAS-len(selected)))
    return selected, len(eligible)


def prepare(project):
    motion = project / "training/molecular-motion"
    source = motion / "rapid-evidence-20261107-v1/reports/checkpoint-audit32/repaired45k"
    names = [name for row in json.loads((source/"metrics.json").read_text())["sample_info"]
             for name in row["names"]][OFFSET:OFFSET+MOLECULES]
    assert len(names) == MOLECULES
    constants = project / "research/molecular-video/mdgen/mdgen/residue_constants.py"
    mapping = next(ast.literal_eval(node.value) for node in ast.parse(constants.read_text()).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                   t.id == "restype_name_to_atom14_names" for t in node.targets))
    folder = ROOT / "inputs"
    folder.mkdir(exist_ok=True)
    write_json(folder/"atom14.json", mapping)
    plan = []
    with np.load(source/"trajectories.npz") as data:
        for i, name in enumerate(names, start=OFFSET):
            original = data["reference"][i, 0].astype(float)
            generated = data["generated"][i].astype(float)
            np.testing.assert_allclose(original, generated[0], atol=1e-5, rtol=0)
            mask = data["atom_mask"][i].astype(bool)
            frames, eligible = choose_candidates(generated, original, mask)
            np.savez_compressed(folder/f"{name}.npz", original=original, model=generated[frames], mask=mask)
            pdb = ROOT/"pdb"/f"{name}.pdb"
            (folder/f"{name}.pdb").write_bytes(pdb.read_bytes())
            plan.append({"name": name, "candidate_frames": frames, "eligible_frames": eligible,
                         "distinct_candidate_frames": len(set(frames)),
                         "seeds": [721000+i*10+j for j in range(REPLICAS)]})
    write_json(folder/"protocol.json", {
        "molecules": plan, "input": "fixed 45k free rollouts, remaining 24 validation names, no future reference for selection",
        "shortage_policy": "When fewer than three frames pass screening, reuse the first selected admissible frame with the next velocity seed",
        "force_field": "AMBER14 and GBn2 implicit solvent, 300 K, pH 7, fixed neutral HIE, 1/ps Langevin friction, HBonds",
        "integration_step_ps": STEP_PS, "equilibration_ps": EQUIL_STEPS*STEP_PS,
        "production_ps_per_replica": SAMPLE_STRIDE*SAMPLES*STEP_PS,
        "production_observation_interval_ps": SAMPLE_STRIDE*STEP_PS,
        "replicas_per_arm_per_molecule": REPLICAS, "primary_bin_width_degrees": 60,
        "sensitivity_bin_width_degrees": 90, "minimum_observations": 3,
    })
    print(json.dumps(plan), flush=True)


def terminal_oxygen(template, positions):
    template_fit = template[[0, 1, 2]]
    p, q = template_fit-template_fit.mean(0), positions-positions.mean(0)
    u, _, vt = np.linalg.svd(p.T@q)
    u[:, -1] *= np.linalg.det(u@vt)
    return (template[3]-template_fit.mean(0))@(u@vt)+positions.mean(0)


def simulate(case):
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation

    name, arm, replicate, seed = case
    folder = ROOT/"inputs"
    with np.load(folder/f"{name}.npz") as data:
        coordinates = data[arm] if arm == "original" else data[arm][replicate]
        mask = data["mask"]
    mapping = json.loads((folder/"atom14.json").read_text())
    pdb = PDBFile(str(folder/f"{name}.pdb"))
    template = np.array(pdb.positions.value_in_unit(unit.angstrom))
    residues = list(pdb.topology.residues())
    assert len(residues) == 4
    locations = {}
    atoms = list(pdb.topology.atoms())
    for i, residue in enumerate(residues):
        for atom in residue.atoms():
            if atom.name != "OXT":
                slot = mapping[residue.name].index(atom.name)
                assert mask[i, slot]
                locations[(i, atom.name)] = atom.index
    oxt = [a for a in atoms if a.name == "OXT"]
    assert len(oxt) == 1 and oxt[0].residue.index == 3
    positions = template.copy()
    for (res, atom), idx in locations.items():
        positions[idx] = coordinates[res, mapping[residues[res].name].index(atom)]
    end = [locations[3, atom] for atom in ("CA", "C", "O")]
    positions[oxt[0].index] = terminal_oxygen(template[end+[oxt[0].index]], positions[end])
    modeller = Modeller(pdb.topology, positions*unit.angstrom)
    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    platform = Platform.getPlatformByName("CPU")
    platform.setPropertyDefaultValue("Threads", "1")
    random.seed(seed)
    variants = modeller.addHydrogens(ff, pH=7.0,
                                   variants=["HIE" if r.name == "HIS" else None for r in residues],
                                   platform=platform)
    topology_key = "|".join(f"{a.residue.index}:{a.residue.name}:{a.name}" for a in modeller.topology.atoms())
    system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, STEP_PS*unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    sim = Simulation(modeller.topology, system, integrator, platform, {"Threads": "1"})
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(300*unit.kelvin, seed+100000)
    indices = [(r.index, mapping[r.name].index(a.name), a.index) for r in modeller.topology.residues()
               for a in r.atoms() if a.name in mapping[r.name] and a.name]
    res_idx, slot_idx, atom_idx = np.array(indices).T
    assert len(atom_idx) == int(mask.sum())

    def snapshot():
        positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        result = np.zeros((4, 14, 3))
        result[res_idx, slot_idx] = positions[atom_idx]
        return result

    sim.step(EQUIL_STEPS)
    equilibrated = snapshot()
    frames = []
    for _ in range(SAMPLES):
        sim.step(SAMPLE_STRIDE)
        frames.append(snapshot())
    assert sim.currentStep == EQUIL_STEPS+SAMPLE_STRIDE*SAMPLES
    assert np.isfinite(equilibrated).all() and np.isfinite(frames).all()
    final_energy = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert np.isfinite(final_energy)
    output = ROOT/"runs"/f"{name}-{arm}-{replicate}.npz"
    temp = output.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, initial=coordinates, equilibrated=equilibrated, production=np.stack(frames),
                            production_steps=SAMPLE_STRIDE*SAMPLES, seed=seed, energy=final_energy,
                            topology=topology_key, variants=json.dumps(variants))
    temp.replace(output)
    return output.name


def count_bins(frames, width):
    degrees = np.rad2deg(torsions(frames))
    return [tuple(row) for row in (np.floor((degrees+180)/width).astype(int) % (360//width))]


def analyze(plan):
    rows = []
    for spec in plan["molecules"]:
        name = spec["name"]
        with np.load(ROOT/"inputs"/f"{name}.npz") as data:
            mask = data["mask"]
        starts, frames, rates = [], {}, {}
        topology = None
        for arm in ARMS:
            samples = []
            for i, seed in enumerate(spec["seeds"]):
                with np.load(ROOT/"runs"/f"{name}-{arm}-{i}.npz") as data:
                    assert data["production_steps"] == SAMPLE_STRIDE*SAMPLES and data["seed"] == seed
                    current = (str(data["topology"]), str(data["variants"]))
                    if topology is None:
                        topology = current
                    assert topology == current, name
                    starts.extend((data["initial"], data["equilibrated"]))
                    samples.append(data["production"])
            production = np.concatenate(samples)
            good = valid_geometry(production, mask)
            rates[arm] = float(good.mean())
            frames[arm] = production[good]
        row = {"name": name, "valid_fraction": rates, "bins": {}}
        for width in (60, 90):
            starting_bins = set(count_bins(np.stack(starts), width))
            occupied = {arm: {bin_id for bin_id,n in Counter(count_bins(frames[arm], width)).items() if n >= 3}
                        for arm in ARMS}
            row["bins"][str(width)] = {
                arm: {"visited": len(occupied[arm]), "outside_starting_bins": len(occupied[arm]-starting_bins)}
                for arm in ARMS
            }
        rows.append(row)
    result = {"protocol": plan, "summary": summarize(rows), "per_molecule": rows,
              "mean_geometry_valid_fraction": {arm: float(np.mean([r["valid_fraction"][arm] for r in rows])) for arm in ARMS}}
    write_json(ROOT/"results.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


def summarize(rows):
    draws = np.random.default_rng(20261107).integers(len(rows), size=(10000, len(rows)))
    summary = {}
    for width in ("60", "90"):
        summary[width] = {}
        for metric in ("visited", "outside_starting_bins"):
            pair = {arm: np.array([row["bins"][width][arm][metric] for row in rows]) for arm in ARMS}
            delta = pair["model"]-pair["original"]
            summary[width][metric] = {"mean": {arm: float(x.mean()) for arm,x in pair.items()},
                "delta": float(delta.mean()), "ci95": np.quantile(delta[draws].mean(1), [.025,.975]).tolist(),
                "wins": int((delta>0).sum()), "ties": int((delta==0).sum()), "losses": int((delta<0).sum())}
    return summary


def combine(plan):
    old = json.loads((ROOT.parent/"seeded-md-pilot-20261107-v1/results.json").read_text())
    new = json.loads((ROOT/"results.json").read_text())
    for key in ("force_field", "integration_step_ps", "equilibration_ps", "production_ps_per_replica",
                "production_observation_interval_ps", "replicas_per_arm_per_molecule", "primary_bin_width_degrees",
                "sensitivity_bin_width_degrees", "minimum_observations"):
        assert old["protocol"][key] == plan[key], key
    a, b = old["per_molecule"], new["per_molecule"]
    assert len(a) == 8 and len(b) == MOLECULES and len({r["name"] for r in a+b}) == 32
    assert not {s for r in old["protocol"]["molecules"] for s in r["seeds"]} & {
        s for r in plan["molecules"] for s in r["seeds"]}
    reused = [row["name"] for row in plan["molecules"] if row["distinct_candidate_frames"] < REPLICAS]
    result = {
        "pilot_8": {"n": len(a), "summary": summarize(a)},
        "replication_24": {"n": len(b), "summary": summarize(b)},
        "combined_32": {"n": len(a+b), "summary": summarize(a+b)},
        "replication_excluding_reused_starts": {
            "n": len(b)-len(reused), "summary": summarize([r for r in b if r["name"] not in reused])},
        "reused_candidate_molecules": reused,
        "replication_valid_fraction": new["mean_geometry_valid_fraction"],
        "replication_production_ns_per_arm": len(b)*REPLICAS*SAMPLE_STRIDE*SAMPLES*STEP_PS/1000,
        "combined_production_ns_per_arm": len(a+b)*REPLICAS*SAMPLE_STRIDE*SAMPLES*STEP_PS/1000,
    }
    write_json(ROOT/"comparison.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
        return
    plan = json.loads((ROOT/"inputs/protocol.json").read_text())
    cases = [(r["name"], arm, i, seed) for r in plan["molecules"]
             for i, seed in enumerate(r["seeds"]) for arm in ARMS]
    (ROOT/"runs").mkdir(exist_ok=True)
    if args.smoke:
        print(simulate(cases[0]), flush=True)
        print(simulate(cases[1]), flush=True)
        return
    remaining = [c for c in cases if not (ROOT/"runs"/f"{c[0]}-{c[1]}-{c[2]}.npz").is_file()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, name in enumerate(pool.map(simulate, remaining), start=len(cases)-len(remaining)+1):
            print(f"{done}/{len(cases)} {name}", flush=True)
    analyze(plan)
    combine(plan)


if __name__ == "__main__":
    main()
