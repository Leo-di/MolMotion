"""Synthetic contract check, NOT a molecular simulation or quality benchmark."""
import tempfile
from pathlib import Path
import numpy as np
from molmotion import evaluate_npz

with tempfile.TemporaryDirectory() as directory:
    x=np.random.default_rng(7).normal(size=(2,17,4,14,3))
    p=Path(directory)/"synthetic.npz"
    np.savez(p,generated=x,reference=x.copy(),sequence_ids=np.zeros((2,4),dtype=int),atom_mask=np.ones((2,4,14)))
    result=evaluate_npz(p)
    assert result["velocity_sliced_w1"] < 1e-10
    assert result["acf_mae"] < 1e-10
    print("安装检查通过：同一输入的分布误差为零。合成数据不用于科学结论。")
