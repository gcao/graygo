# Independently audited RPS tabular control

The seed-0 primary control completed **15M aggregate own-side records** and passed all final-five original-game average-policy gaps. Independent reconstruction verifies all archived policies, the closed-form game value zero, both private mode/ramp streams, action draw counts, exact Algorithm R replacement/version history, replay records and optimizer/refresh counters. The worst final-window gap is 0.023302; the recorded trial wall time is 1341.001 seconds.

The source is the same pinned matrix implementation as the asymmetric trials: commit `6a92132319c755fc6b1f8b1b5e4ebea8e75e1e98`, SHA `4c13ba875326ed1f843cb3ca31433063e72793cb8a677d4968a0652766eb3e0c`. Its 5M opening, 5M–7M ramp, epsilon clock, native mode mix and full record cap are unchanged. Raw audit/run metadata are preserved beside this report; complete local evidence remains in `runs/nfsp-rps-primary-6a92132/trials/tabular` on `ai`.

The initial auditor reproduced the three-action uniform arithmetic in a different float32 operation order, producing a probability discrepancy around 7e-9 at an archived boundary. The corrected audit divides the uniform distribution before multiplying epsilon, as the actual sampler does. Tolerances, policy gaps and training were not changed. New RPS neural/tabular regression traces cover that path.

The RPS neural and delayed first screens remain live under their original handles. This control result alone does not complete RPS, the three-seed cohort, A2, or deployment readiness.
