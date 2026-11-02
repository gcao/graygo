# Recorded experiment resources

These are completed sequential **CPU process wall times**, not accelerator-hours or a complete development-cost total. Raw state artifacts on `ai` supply each value. Tests, independent audit runtime and small engine/synthetic-network profiles are additional overhead not rolled into this table; do not call its sum complete project consumption.

| Completed component | Seconds | CPU process wall hours |
|---|---:|---:|
| PPO matrix preflight | 1082.360 | 0.30066 |
| NFSP terminal-matrix timing | 38.253 | 0.01063 |
| Asymmetric NFSP first screen | 2776.246 | 0.77118 |
| Historical delayed timing | 232.936 | 0.06470 |
| Corrected delayed timing | 229.264 | 0.06368 |
| RPS NFSP first screen (both completed cases) | 2782.993 | 0.77305 |
| Delayed R1 first screen (completed tabular case) | 7218.411 | 2.00511 |
| Delayed R1 first screen (completed neural case) | 7354.825 | 2.04301 |

The historical delayed cost is retained even though its source cannot fund the corrected evaluator. The corrected cost report enters C1 and the separate tabular-control ceiling; C4 remains conditional. RPS is complete and independently audited. The delayed tabular case completed the full 15M records and reports a gate miss; its independent task audit now confirms the grade. Its matching neural case is also complete and independently audited; its average numerical gap passes, while supported credit misses. R1 delayed first-screen trial settings are bound to the measured budget hashes before launch. H_P, H_D, H_N, CE and the full-study envelope remain unfilled until their applicable selection/profiling steps. CPU terminal/opaque-task costs never substitute for 9×9 accelerator measurements.


## Bounded accelerator pipeline measurement

The full-model 9x9 R1 hardware exercise (`dc97531`, `verification/board-runtime.md`) completed two 16,384-record phases in **50.966 seconds**, or **0.01416 elapsed hours of a GPU-equipped process**. This is an allocated-process wall time, not measured kernel-active GPU time. It includes collection/learner setup, full-network Q/target/averaging work, strict checkpoint write/restore, and 16 small timing-only evaluation games. The CUDA allocation peak was 830,899,200 bytes; reservation peaked at 1,042,284,544 bytes. The original profile process is terminal, with audited final evidence.

The early-open phase is an explicit diagnostic. Untrained game lengths, partial buffer populations and tiny evaluation samples do not establish H_P/H_D/H_N. The independent full-capacity storage profile and previous CPU experiments remain separate measured costs; tests/audits/builds/development overhead are still not comprehensively summed. No large campaign allocation follows from this measurement.


## Candidate-independent probe construction

The preserved feasibility diagnostics consumed 47.908 seconds (3x3 recoloring), 28.071 seconds (5x5 dense), 0.242 seconds (5x5 liberty change) and 0.400 seconds (9x9 liberty change), as process wall times. The source-bound reachable catalog construction then consumed 4.915 seconds. Subsequent proof/lineage verification and tests are additional overhead, and exploratory setup checks are not comprehensively summed here. These are CPU construction measurements under the separate eight-hour probe allocation, not accelerator campaign prices.

The generated catalog has 32 geometric pairs but one complete-game strategic class. Family coverage is incomplete; the 24-of-32 gate cannot pass on these repeated instances. The failed/limited diagnostics remain visible instead of being discarded. No trained policy, performance endpoint or gray-specialization outcome was measured by these construction jobs.


## First R4 delayed allocation

The frozen plan in `nfsp-delayed-R4-primary-plan.json` inventories fifteen completed tabular control/profile leaf runs by manifest/result/final-checkpoint hashes. Their 10,158.735 process-wall seconds include unused historical profiles. The single first-screen R4 tabular case reserves 18,900 seconds, for 29,058.735 seconds total (8.07187 hours) within the twelve-hour experimental-control ceiling. No remaining cohort is silently funded. Both R4 cases use CPU; the neural case has its own 18,900-second matched-profile ceiling. GPU/campaign prices are not inferred from these numbers.


## Scientific-evaluator integration diagnostic

The matched collector completed 384 random-baseline games across one resumed and one uninterrupted small diagnostic. Observed setup/collection clocks were 6.211 and 5.281 seconds; both had zero recovery-gap charge. Startup, read-only analysis, tests and development overhead are not comprehensively summed here. This CPU integration check is separate from the unselected 24k/96k confirmation evaluation and its 16-device-hour scientific/probe-inference envelope. Its source, full scored evidence and exact-resume audit are in `verification/scientific-contrast.md`.


## Bounded board milestone recovery diagnostic

The clean `546d2e2` full-model CUDA lifecycle exercise completed two 16,384-record trajectories (32,768 aggregate fresh records) in **87.245 process-wall seconds**, or **0.02423 elapsed hours of a GPU-equipped process**. It includes full model updates, archive/checkpoint IO, public loading and interrupted continuation. This is allocated process time, not measured kernel-active GPU time. Its independent artifact audit and test/development overhead are additional and are not comprehensively summed here.

The [frozen evidence](../verification/board-milestone-diagnostic/README.md) uses reference Python decoding, Q reuse four and reduced diagnostic opening/hand-off clocks. It is not a reuse-only comparison with the earlier R1/native profile, a full-population measurement, or H_P/H_D/H_N/campaign pricing. No campaign allocation follows from this diagnostic.
