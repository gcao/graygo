# Corrected R1 complete-learner pricing and independent audit

The four bounded matched cases are complete under clean frozen source `ca45a53`, package SHA `db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873`. Original profile PID **2439426** is terminal/absent. The [full copied artifacts](nfsp-eligibility-R1-profiles/README.md) retain the original case manifests/results, exact profiles, budget files, model/checkpoint inventories and all four independent corrected-source audits.

| Learner/phase | Aggregate records | Eligible Q records | Seconds incl. setup | Q updates | Target copies |
|---|---:|---:|---:|---:|---:|
| Tabular pre | 65,536 | 65,461 | 34.239 | 120 | 5 |
| Tabular post | 65,536 | 65,461 | 31.735 | 120 | 5 |
| Neural pre | 65,536 | 65,461 | 34.067 | 120 | 5 |
| Neural post | 65,536 | 65,461 | 32.822 | 120 | 5 |

Every case uses the production shared target/loss/replay/cadence and actual opaque 50/100/200-round, role-balanced 8/1/1 coverage with forty equivalent continuation actions. Only the post diagnostic opens at zero and ends its hand-off at 1,000 to exercise averaging cheaply. All cases contain real Q/target, replay, evaluation and checkpoint work; both post cases include supervised work and learned-average acting. Independent source-pinned audits verify masks/clocks/roles, coverage, Algorithm-R, private sampler RNG, retained transitions/labels and closed-form native values. None is a convergence or A2 pass.

The fixed 5M-pre/10M-post extrapolation forecasts **7,454.558 CPU seconds** tabular and **7,607.401** neural per 15M-record case. Each proposes a **15,300-second** ceiling, with margins **2.0524** and **2.0112**. Partially populated buffers, reduced diagnostic hand-off and scaled startup/evaluation/IO overhead remain limitations. This prices corrected opaque-chain R1 only, never 9x9, R4 or Retrace. Audits and broader development are additional overhead.

The [profile settlement](../experiments/nfsp-eligibility-R1-profile-settlement.json) records 132.863 measured case seconds including setup (driver interval 133.379). The two tabular profiles add 65.973 seconds, giving eighteen recorded control/profile leaves and 19,475.063 seconds consumed within the twelve-hour experimental-control ceiling. No other allocation is inferred from those measurements.

The separately recorded [corrective first-screen allocation](../experiments/nfsp-eligibility-correction-R1-allocation.json) funds only a fresh seed-0 tabular/neural R1 pair under these source-bound caps. No later opening, target variant, extra cohort or full board campaign is selected.
