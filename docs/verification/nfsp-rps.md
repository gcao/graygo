# Audited RPS first screen

Both seed-0 primary trials completed 15M aggregate own-side records and independently passed all final-five original-game average-policy gaps. The auditor reconstructs every archived model and original security bound, and checks episode choices, independent RNG streams, Algorithm R, retained transitions and update/refresh counters. The original stage is terminal; it was not restarted.

| Learner | Trial seconds | Worst final-five gap | Independent pass |
|---|---:|---:|---|
| tabular | 1341.001 | 0.023302 | True |
| neural | 1441.992 | 0.012268 | True |

Stage wall time including its cost profiles: 2821.233 seconds. Both cases retain source commit `6a92132319c755fc6b1f8b1b5e4ebea8e75e1e98`, SHA `4c13ba875326ed1f843cb3ca31433063e72793cb8a677d4968a0652766eb3e0c`.

The original 5M opening, 5M–7M hand-off, fixed epsilon clock and full aggregate record cap remain unchanged. Tabular and neural controls each account for 7.5M records per role. Raw audit/run/plan/cost artifacts are preserved beside this report. This is one seed of two matrix families; the delayed first screen is still live and bounded/branching and remaining-seed evidence are not waived. No 9×9 pilot or scientific conclusion is authorized by these passes.
