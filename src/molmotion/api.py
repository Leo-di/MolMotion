"""Coordinate-based trajectory evaluation; does not require PyTorch."""
from pathlib import Path
import numpy as np
from .metrics import metrics


def evaluate_npz(path):
    """Evaluate B,T,4,14,3 atom14 coordinates in Angstrom.

    Required fields: generated, reference, sequence_ids, atom_mask.
    Velocity W1 uses displacement per frame; inputs must share frame spacing.
    This is a pooled evaluation, not a molecule-wise significance test.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        required = ("generated", "reference", "sequence_ids", "atom_mask")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError("Missing fields: " + ", ".join(missing))
        x, y, seq, mask = (data[k] for k in required)
    if x.ndim != 5 or x.shape[2:] != (4,14,3) or x.shape != y.shape or x.shape[0] < 1 or x.shape[1] < 3:
        raise ValueError("Expected matching [B,T,4,14,3] coordinates with B>=1, T>=3")
    if seq.shape != (len(x),4) or mask.shape != (len(x),4,14):
        raise ValueError("Expected sequence_ids [B,4] and atom_mask [B,4,14]")
    if not all(np.isfinite(v).all() for v in (x,y,mask)):
        raise ValueError("Coordinates and mask must be finite")
    if not np.isin(mask, (0,1)).all() or not mask[:,:,:4].all():
        raise ValueError("Binary atom mask must include all four backbone atoms")
    values, _ = metrics(x.astype(float), y.astype(float), seq, mask)
    return values
