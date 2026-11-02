# Audited asymmetric NFSP first screen

Both seed-0 primary trials completed 15M aggregate own-side records and passed the original-game average-policy gap criterion at every one of the final five evaluations. The archive-based independent audit verifies this outcome; it is one task/seed, not completion of A2.

Source commit: `6a92132319c755fc6b1f8b1b5e4ebea8e75e1e98`; source SHA-256: `4c13ba875326ed1f843cb3ca31433063e72793cb8a677d4968a0652766eb3e0c`. Both trials use the unchanged 5M opening, 5M–7M hand-off, fixed A2 epsilon clock, R=1 and production matrix update/replay paths. Each has 7.5M records per role, 117,156 Q updates and 78,114 supervised updates/refreshes.

| Learner | Recorded trial seconds | Worst gap in final five | Independently verified pass |
|---|---:|---:|---|
| tabular | 1338.745 | 0.037677 | True |
| neural | 1436.630 | 0.027430 | True |

The sequential first-screen driver consumed 2776.246 seconds of CPU process wall time (0.7712 hours). GPU training was not used. The original process is terminal; do not infer status solely from its historical PID.

The auditor reconstructs all 30 scheduled policies from checksummed public weights, uses the closed-form game value `-1/7`, recomputes security gaps, verifies both independent mode/ramp streams and the private action draw counts, reconstructs Algorithm R selection and label-version records, checks optimizer/fresh/refresh counters, and compares every retained executed transition with the permanent episode log. It also checks actual behavior probabilities/actions at archived-model boundaries. The tabular deployed policy is independently compared with the full-reservoir label mean. Its unit checks include replay wraparound, reservoir replacement, altered result gates and corrupted episode bytes.

Raw audit reports and run manifests are preserved beside this report. The full local weight/log/buffer evidence remains under `runs/nfsp-asymmetric-primary-6a92132` on `ai`. Run the auditor with the original worktree on `PYTHONPATH`:

```sh
PYTHONPATH=.worktrees/nfsp-matrix-6a92132/src .venv/bin/python tools/audit_nfsp_matrix.py runs/nfsp-asymmetric-primary-6a92132/neural --output /tmp/neural-audit.json
```

RPS, delayed/branching controls, bounded Go and the required additional seeds remain mandatory. A passing matrix average is not evidence of delayed credit or full-board strength.
