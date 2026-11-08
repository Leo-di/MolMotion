# MolMotion

Slow–fast tetrapeptide trajectory generation and simulation-start selection. **Research preview**, not a validated general-purpose molecular simulator.

## What exists now

- Actual hierarchical and single-scale Transformer implementations (`src/molmotion/model.py`).
- Coordinate trajectory metrics (`src/molmotion/metrics.py`): peptide-bond validity, sliced velocity W1, torsional ACF, clashes and related diagnostics.
- Original research training/evaluation runtime and isolated repair variants under `research/`.
- Historical experiment tables and timestamped result snapshots under `results/`.

## Install locally

```sh
python -m pip install .
# Optional neural-network component:
python -m pip install '.[model]'
```

```python
from molmotion.model import HierarchicalDynamics, SingleScaleDynamics, generate
from molmotion.metrics import metrics
```

`generate` currently takes encoded, normalized tensors: initial `[B,4,21]`, relative geometry `[B,4,7]`, sequence `[B,4]`, feature mask `[B,4,21]`, and a compatible trained model. It returns normalized features including the initial frame, slow states and residuals. **These are not atom coordinates**; MDGen geometry decoding and matching normalization are required. See the original runtime for that conversion. No weights are bundled, and randomly initialized output is not scientifically usable.

There is no `from_pretrained`, direct PDB-to-trajectory SDK, or one-call `propose_seeds` implementation in this preview. Do not substitute the proposed API for an existing capability.

## Architecture and scope

Four residues, 21 geometric features per residue, 256 future frames at 10 ps spacing. Eight-frame block means form 32 slow states. A causal 12-layer width-768 Transformer generates slow history; a conditional 16-layer width-768 Transformer generates zero-mean fast residuals. The single-scale comparator has 28 layers. Flow matching is trained with geometry penalties. Inference does not use true future frames. Original training uses true slow-history conditioning; rollout mismatch and slow-branch overfitting are active research issues.

## Reproduction dependencies

`research/runtime` is the original environment-dependent experiment code, not a standalone installed trainer. It needs MDGen and its dependencies, prepared trajectory caches, split CSVs, a READY marker, normalization and checkpoint metadata. Set `MDGEN_REPO`, `V2_DATA`, and `V2_ROOT` for your environment. `configs/training-protocol.json` records the experimental budget. Exact training environments and self-contained commands still need consolidation; installation of the lightweight package alone does not reproduce full training.

## Results

Read [evidence status](docs/STATUS.md) before interpreting CSVs. The single-scale baseline currently leads in bond validity. Slow–fast decomposition has useful diagnostic and candidate-selection evidence, but not an established overall superiority. Equal-simulation-step coverage gains do not imply equal-wall-time gains.

## Release plan

1. Freeze complete analysis data, provenance and matched evaluation protocols.
2. Audit code ownership and third-party notices, then select the original-code license.
3. Package checkpoint loading, geometry conversion and a fixed four-residue example with regression tests.
4. Add `generate`, `evaluate`, `export` and `propose_seeds` workflows; independently host licensed checkpoints.
5. Validate a clean installation before any PyPI release. REST/gRPC/MCP services can follow later.

See [third-party notices](THIRD_PARTY_NOTICES.md). No PyPI or model-hub release has been made.
