# Evidence status — 2026-11-08 morning snapshot

- Completed: hierarchical seeds 43/44 and single-scale seeds 43/45, 100k training. Selected-checkpoint final evaluations use 128 clips from 32 validation molecules; they are not 128 independent molecules or a held-out test.
- Single-scale seed 44 failed near 9.4k steps. Its paired 150k continuation is cancelled; do not present three completed paired seeds.
- Fixed 25k/65k evaluations use 32 molecules. At 65k hierarchical bond validity is 0.873379, single-scale 0.952943. Hierarchical minus baseline paired CI is [-0.124637, -0.043896].
- At 65k pooled velocity sliced W1 is 0.023200 vs 0.024433, but mean molecule-wise difference is +0.006142 (CI [-0.000528, 0.014523]). Pooled and molecule-wise aggregations differ; a consistent molecule-wise gain is not established.
- Equal MD steps: 32 tetrapeptides, 3 x 50 ps per arm; excluding starting bins, 60-degree-bin new coverage improves by 2.6875 on average (paired CI [1.09375, 4.1875]), 24/32 wins.
- Equal end-to-end CPU wall time: generation exceeds the short reference budget, leaving zero production MD for the model arm. This implementation has not demonstrated end-to-end wall-time benefit.
- Low-LR 40k matched-eight pilot: bond +1.41 percentage points, velocity W1 -12.1%; ACF and clashes worsen. Full repair comparisons remain pending.
- LaM-SLidE finished but has invalid geometry (~4.76% bond validity). Adapter auditing is required before a fair method ranking.
- Dropout, geometry-detach, low-LR paired final evaluations, TEMPO final results, and independent test-set evaluation are not completed in this snapshot.

Historical CSV files retain their original evaluation protocols; never pool rows across horizons, clips, checkpoints, or budgets without matching those fields. JSON inventories contain stale status files as historical evidence, not live scheduler truth.
