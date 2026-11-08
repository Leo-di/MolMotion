"""OpenMM setup reused from the paired seed-selection pilot."""
import json
import random
import numpy as np

from pilot import STEP_PS, terminal_oxygen


def setup_simulation(folder, name, coordinates, mask, seed, threads):
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation

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
    platform.setPropertyDefaultValue("Threads", str(threads))
    random.seed(seed)
    variants = modeller.addHydrogens(ff, pH=7.0,
                                   variants=["HIE" if r.name == "HIS" else None for r in residues],
                                   platform=platform)
    topology_key = "|".join(f"{a.residue.index}:{a.residue.name}:{a.name}" for a in modeller.topology.atoms())
    system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, STEP_PS*unit.picoseconds)
    integrator.setRandomNumberSeed(seed)
    sim = Simulation(modeller.topology, system, integrator, platform, {"Threads": str(threads)})
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

    return sim, snapshot, (topology_key, tuple(variants))
