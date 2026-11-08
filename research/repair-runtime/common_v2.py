"""Shared data and geometry functions for the sealed second-round experiments."""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(os.environ.get("V2_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("V2_DATA", "/base/data/processed"))
REPO = Path(os.environ.get("MDGEN_REPO", "/base/repos/mdgen"))
sys.path.insert(0, str(REPO))

from mdgen.parsing import parse_train_args
from mdgen.rigid_utils import Rigid, Rotation
from mdgen.utils import get_offsets
from mdgen.model.latent_model import LatentMDGenModel
from mdgen.transport.transport import create_transport, Sampler
from mdgen.geometry import frames_torsions_to_atom14
from mdgen import residue_constants as rc


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arguments(frames=64, width=256, layers=4):
    original = sys.argv
    sys.argv = ["mdgen", "--train_split", str(DATA / "train.csv"),
                "--val_split", str(DATA / "val.csv"), "--data_dir", str(DATA),
                "--num_frames", str(frames), "--crop", "4", "--sim_condition",
                "--prepend_ipa", "--abs_pos_emb", "--abs_time_emb",
                "--embed_dim", str(width), "--num_layers", str(layers),
                "--mha_heads", "8"]
    try:
        return parse_train_args()
    finally:
        sys.argv = original


def zero_temporal_output(module, inputs, output):
    return torch.zeros_like(output)


def make_model(args, mode, device):
    model = LatentMDGenModel(args, 21).to(device)
    if mode == "framewise":
        for layer in model.layers:
            layer.mha_t.register_forward_hook(zero_temporal_output)
    return model


class Trajectories:
    def __init__(self, split, device, limit=None):
        assert (DATA / "READY").exists(), DATA
        names = [line.split(",")[0] for line in (DATA / f"{split}.csv").read_text().splitlines()[1:]]
        self.names = names if limit is None else names[:limit]
        self.device = device
        self.arrays = {}
        for name in self.names:
            with np.load(DATA / f"{name}_cache.npz") as data:
                self.arrays[name] = {key: torch.tensor(data[key], device="cpu") for key in data.files}

    def batch(self, batch_size, frames, seed, shuffle=False, balanced=False, name_offset=0):
        rng = np.random.default_rng(seed)
        rows, names, starts = [], [], []
        for index in range(batch_size):
            name = self.names[(index + name_offset) % len(self.names)] if balanced else self.names[int(rng.integers(len(self.names)))]
            data = self.arrays[name]
            start = int(rng.integers(len(data["trans"]) - frames + 1))
            order = np.arange(frames)
            if shuffle:
                order[1:] = rng.permutation(order[1:])
            indices = torch.tensor(order + start, device="cpu")
            rows.append({"trans": data["trans"][indices], "rots": data["rots"][indices],
                         "torsions": data["torsions"][indices], "seqres": data["seqres"],
                         "torsion_mask": data["torsion_mask"],
                         "mask": torch.ones(4, device=self.device)})
            names.append(name)
            starts.append(start)
        batch = {key: torch.stack([row[key] for row in rows]).to(self.device) for key in rows[0]}
        return batch, {"names": names, "starts": starts, "shuffled": shuffle}


def prepare_batch(batch):
    rigids = Rigid(trans=batch["trans"], rots=Rotation(rot_mats=batch["rots"]))
    batch_size, frames, residues = rigids.shape
    offsets = get_offsets(rigids[:, 0:1], rigids)
    offsets[..., :4] *= torch.where(offsets[..., 0:1] < 0, -1, 1)
    latents = torch.cat([offsets, batch["torsions"].reshape(batch_size, frames, residues, 14)], -1)
    frame_mask = batch["mask"].unsqueeze(-1).expand(-1, -1, 7)
    torsion_mask = batch["torsion_mask"].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(batch_size, residues, 14)
    loss_mask = torch.cat([frame_mask, torsion_mask], -1).unsqueeze(1).expand(-1, frames, -1, -1)
    condition_mask = torch.zeros(batch_size, frames, residues, dtype=torch.long, device=latents.device)
    condition_mask[:, 0] = 1
    return {"latents": latents, "loss_mask": loss_mask, "rigids": rigids,
            "model_kwargs": {"start_frames": rigids[:, 0], "end_frames": rigids[:, -1],
                             "mask": batch["mask"].unsqueeze(1).expand(-1, frames, -1),
                             "aatype": batch["seqres"], "x_cond_mask": condition_mask,
                             "x_cond": torch.where(condition_mask.unsqueeze(-1).bool(), latents, 0.0)}}


def decode_latents(latents, prepared, sequence):
    offsets = latents[..., :7]
    frames = prepared["rigids"][:, 0:1].compose(Rigid.from_tensor_7(offsets, normalize_quats=True))
    torsions = latents[..., 7:21].reshape(*latents.shape[:3], 7, 2)
    torsions = torsions / torch.linalg.vector_norm(torsions, dim=-1, keepdim=True).clamp_min(1e-8)
    atom14 = frames_torsions_to_atom14(frames, torsions, sequence[:, None].expand(*latents.shape[:3]))
    return atom14, torsions


def true_atom14(batch, prepared):
    shape = batch["torsions"].shape[:3]
    return frames_torsions_to_atom14(prepared["rigids"], batch["torsions"],
                                     batch["seqres"][:, None].expand(*shape))


def atom14_mask(sequence, frames):
    mask = torch.tensor(rc.RESTYPE_ATOM14_MASK, device=sequence.device)[sequence]
    return mask[:, None].expand(-1, frames, -1, -1)


def verify_inputs():
    fingerprints = json.loads((DATA / "fingerprints.json").read_text())
    for name, expected in fingerprints.items():
        assert sha(DATA / name) == expected, name
    return fingerprints


@torch.no_grad()
def sample_flow(model, transport, prepared, seed, evaluations=64):
    torch.manual_seed(seed)
    noise = torch.randn_like(prepared["latents"])
    sampler = Sampler(transport).sample_ode(sampling_method="euler", num_steps=evaluations)
    generated = sampler(noise, model.forward_inference, **prepared["model_kwargs"])[-1]
    condition = prepared["model_kwargs"]["x_cond_mask"].unsqueeze(-1).bool()
    return torch.where(condition, prepared["latents"], generated)
