# NFSP terminal-matrix cost profile

Four sequential CPU profiles exercised native self-play, production Q loss/replay, supervised averaging, per-episode audit logging, exact evaluation and checksummed checkpoint IO. Each used 100,000 fresh records (50,000 physical games). The post-opening timing diagnostic opens at zero and completes its ramp at 1,000 records to exercise the average policy; it is explicitly not a formal training candidate.

Source commit: `6a92132319c755fc6b1f8b1b5e4ebea8e75e1e98`. Source SHA-256: `4c13ba875326ed1f843cb3ca31433063e72793cb8a677d4968a0652766eb3e0c`. Pinned worktree: `.worktrees/nfsp-matrix-6a92132`. All raw plans/manifests and the exact profiling driver are preserved beside this report.

| Phase | Learner | Seconds | Fresh records/second | Q updates | Supervised refreshes/updates |
|---|---|---:|---:|---:|---:|
| pre | neural | 8.991 | 11122 | 750 | 0 |
| pre | tabular | 8.700 | 11494 | 750 | 0 |
| post | neural | 10.031 | 9969 | 750 | 771 |
| post | tabular | 9.387 | 10653 | 750 | 771 |

The pre-opening profiles used Q in both logical modes and collected no labels. The post-opening cases used the learned average actor for 88,768 of 100,000 records, exercising the selected actor mode mix, while collecting 9,955 best-response labels. A tabular supervised refresh publishes the exact reservoir mean rather than performing a gradient update. Q sampling uses an independent stream in both learners.

Each buffer was allocated for 200,000 rows, but replay had 100,000 filled rows and the post-opening reservoir had 9,955. Logical array allocations were 14,200,000 replay bytes and 3,800,000 reservoir bytes; these are not resident-memory measurements or a full-capacity checkpoint profile. Episode evidence occupies 121 bytes per physical game: 6,050,000 bytes per profile and a projected 907,500,000 bytes for one complete 15M-record matrix trial.

## First asymmetric screen budget

For each learner, a simple phase-weighted projection is `50*pre_profile_seconds + 100*post_profile_seconds` (5M pre-opening + 10M later records). It does not model every transient during hand-off or larger audit-prefix hashing/checkpoints.
- tabular: 1373.7 seconds (22.9 minutes); preregistered ceiling 3,600 seconds, exceeding twice the projection.
- neural: 1452.7 seconds (24.2 minutes); preregistered ceiling 3,600 seconds, exceeding twice the projection.

The first screen runs asymmetric seed 0, tabular control followed by neural learner, at the unchanged full 15M-record primary recipe. Its combined ceiling is two CPU process-hours. It does not fund a multiround or 9×9 campaign. The run plan is tracked in `../experiments/nfsp-asymmetric-primary-plan.json`; its source worktree and driver are fixed. Actual elapsed cost, rather than the ceiling, will enter the cumulative ledger.

No convergence result follows from these timing profiles. RPS, bounded-game target inference, opaque delayed/branching tasks, full-capacity IO and all 9×9 components need their own evidence before the corresponding A2-exit/full-study budget decision. In particular `C1`, `C4`, `CT`, `H_P`, `H_D` and `H_N` remain unfilled; no full campaign is priced from this terminal-only measurement.
