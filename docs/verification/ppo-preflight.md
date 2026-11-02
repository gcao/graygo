# Formal PPO matrix preflight

Both prescribed PPO protocols failed route selection. The selected continuation is NFSP; this finding applies to these recipes and does not establish failure of PPO in general.

- Source commit: `1cd4e0d12913b2bf63c8883d30e8a3d71f3c431e`.
- Source SHA-256: `32f82bd769cf56af417977d3638bed7484094ca123faa79f7586c6b6788ae10d`.
- Twelve complete CPU trials; 2M learner decisions per trial, balanced 1M per role; 24M aggregate learner decisions.
- Recorded sequential preflight wall time: 1082.36 seconds. This is a matrix-task measurement, not a 9×9 learner estimate.
- Both roles must have gap ≤0.05 at each of the final five scheduled evaluations. A single protocol and deployed representation must pass both games and all three seeds.
- Mixtures select each player's checkpoint independently once per episode, uniformly over scheduled evaluations in the latter half. For one-round games, averaging probabilities computes this episode mixture exactly.

| Protocol | Game | Seed | Final latest max gap | Final mixture max gap | Latest final-five pass | Mixture final-five pass |
|---|---|---:|---:|---:|---|---|
| fixed-0.1 | asymmetric | 0 | 0.138867 | 0.053304 | False | False |
| fixed-0.1 | asymmetric | 1 | 0.076328 | 0.027315 | False | True |
| fixed-0.1 | asymmetric | 2 | 0.088536 | 0.006106 | False | True |
| taper | asymmetric | 0 | 0.848551 | 0.130438 | False | False |
| taper | asymmetric | 1 | 0.854251 | 0.023493 | False | False |
| taper | asymmetric | 2 | 0.807019 | 0.674058 | False | False |
| fixed-0.1 | rps | 0 | 0.062973 | 0.032679 | False | True |
| fixed-0.1 | rps | 1 | 0.124813 | 0.022234 | False | False |
| fixed-0.1 | rps | 2 | 0.055192 | 0.037424 | False | True |
| taper | rps | 0 | 0.999955 | 0.554151 | False | False |
| taper | rps | 1 | 0.999949 | 0.454579 | False | False |
| taper | rps | 2 | 0.999962 | 0.636314 | False | False |

Passing a final snapshot alone is insufficient; the final-five columns determine the gate. No representation passed the complete protocol.

The independent audit verifies checksummed final resume bundles and reconstructs every archived evaluation model, using closed-form game values and independently recomputed security gaps. It checks record/update counts and reconstructs all checkpoint mixtures. The exact-BR tabular reference controls passed separately; they are not the still-pending tabular NFSP Q-learning control.

Artifacts on `ai`: `/home/gcao/proj/graygo/runs/ppo-preflight-1cd4e0d1`. The pinned worktree is `.worktrees/ppo-preflight-1cd4e0d1`. Re-run the auditor using that worktree's `src` on `PYTHONPATH`:

```sh
PYTHONPATH=.worktrees/ppo-preflight-1cd4e0d1/src .venv/bin/python tools/audit_ppo_preflight.py runs/ppo-preflight-1cd4e0d1 --output /tmp/graygo-preflight-audit.json
```

This preflight selects a route only. It provides no delayed-credit, NFSP, 9×9 pilot, or full-game strength result.
