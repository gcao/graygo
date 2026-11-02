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
| Delayed R4 first screen (completed tabular case) | 9250.354 | 2.56954 |
| Delayed R4 first screen (completed neural case) | 9611.440 | 2.66984 |

The historical delayed cost is retained even though its source cannot fund the corrected evaluator. The corrected cost report enters C1 and the separate tabular-control ceiling; C4 remains conditional. RPS is complete and independently audited. The delayed tabular case completed the full 15M records and reports a gate miss; its independent task audit now confirms the grade. Its matching neural case is also complete and independently audited; its average numerical gap passes, while supported credit misses. R1 delayed first-screen trial settings are bound to the measured budget hashes before launch. H_P, H_D, H_N, CE and the full-study envelope remain unfilled until their applicable selection/profiling steps. CPU terminal/opaque-task costs never substitute for 9×9 accelerator measurements.


## Bounded accelerator pipeline measurement

The full-model 9x9 R1 hardware exercise (`dc97531`, `verification/board-runtime.md`) completed two 16,384-record phases in **50.966 seconds**, or **0.01416 elapsed hours of a GPU-equipped process**. This is an allocated-process wall time, not measured kernel-active GPU time. It includes collection/learner setup, full-network Q/target/averaging work, strict checkpoint write/restore, and 16 small timing-only evaluation games. The CUDA allocation peak was 830,899,200 bytes; reservation peaked at 1,042,284,544 bytes. The original profile process is terminal, with audited final evidence.

The early-open phase is an explicit diagnostic. Untrained game lengths, partial buffer populations and tiny evaluation samples do not establish H_P/H_D/H_N. The independent full-capacity storage profile and previous CPU experiments remain separate measured costs; tests/audits/builds/development overhead are still not comprehensively summed. No large campaign allocation follows from this measurement.


## Candidate-independent probe construction

The preserved feasibility diagnostics consumed 47.908 seconds (3x3 recoloring), 28.071 seconds (5x5 dense), 0.242 seconds (5x5 liberty change) and 0.400 seconds (9x9 liberty change), as process wall times. The source-bound reachable catalog construction then consumed 4.915 seconds. Subsequent proof/lineage verification and tests are additional overhead, and exploratory setup checks are not comprehensively summed here. These are CPU construction measurements under the separate eight-hour probe allocation, not accelerator campaign prices.

The generated catalog has 32 geometric pairs but one complete-game strategic class. Family coverage is incomplete; the 24-of-32 gate cannot pass on these repeated instances. The failed/limited diagnostics remain visible instead of being discarded. No trained policy, performance endpoint or gray-specialization outcome was measured by these construction jobs.


## First R4 delayed allocation

The frozen plan in `nfsp-delayed-R4-primary-plan.json` inventories fifteen completed tabular control/profile leaf runs by manifest/result/final-checkpoint hashes. Their 10,158.735 process-wall seconds include unused historical profiles. At launch, the single first-screen R4 tabular case reserved 18,900 seconds, for 29,058.735 seconds total (8.07187 hours) within the twelve-hour experimental-control ceiling. No remaining cohort is silently funded. Both R4 cases use CPU; the neural case has its own 18,900-second matched-profile ceiling. GPU/campaign prices are not inferred from these numbers.


## Scientific-evaluator integration diagnostic

The matched collector completed 384 random-baseline games across one resumed and one uninterrupted small diagnostic. Observed setup/collection clocks were 6.211 and 5.281 seconds; both had zero recovery-gap charge. Startup, read-only analysis, tests and development overhead are not comprehensively summed here. This CPU integration check is separate from the unselected 24k/96k confirmation evaluation and its 16-device-hour scientific/probe-inference envelope. Its source, full scored evidence and exact-resume audit are in `verification/scientific-contrast.md`.


## Bounded board milestone recovery diagnostic

The clean `546d2e2` full-model CUDA lifecycle exercise completed two 16,384-record trajectories (32,768 aggregate fresh records) in **87.245 process-wall seconds**, or **0.02423 elapsed hours of a GPU-equipped process**. It includes full model updates, archive/checkpoint IO, public loading and interrupted continuation. This is allocated process time, not measured kernel-active GPU time. Its independent artifact audit and test/development overhead are additional and are not comprehensively summed here.

The [frozen evidence](../verification/board-milestone-diagnostic/README.md) uses reference Python decoding, Q reuse four and reduced diagnostic opening/hand-off clocks. It is not a reuse-only comparison with the earlier R1/native profile, a full-population measurement, or H_P/H_D/H_N/campaign pricing. No campaign allocation follows from this diagnostic.


## Locked native/CUDA execution integration

The clean `0b15b8c` integration driver completed four actual CLI invocations across two 16,384-record full-model CUDA trajectories in **61.154 process-wall seconds**, or **0.01699 hours of a GPU-equipped process**. The persisted clocks charge 26.021 seconds uninterrupted and 32.286 seconds across the three resumed sessions, with zero recovery-gap charge. These clocks are a breakdown within the driver total. The driver measurement includes command startup/reporting outside the component clocks; its own initial import/setup, independent saved-artifact audit, tests and development are additional overhead.

This is allocated process time, not measured kernel-active GPU time. The [frozen native/CUDA evidence](../verification/board-execution-diagnostic/README.md) uses Q reuse four, full capacity declarations but partial live buffers, untrained game lengths and reduced diagnostic opening/hand-off clocks. It does not price H_P/H_D/H_N or select a development/confirmation campaign. No direct timing comparison isolates Q reuse across the differently configured historical diagnostics.


## R4 completed control settlement and broader probe feasibility

The original seed-0 R4 tabular trial is complete at 15M records and independently audited. Its measured **9,250.354 CPU process-wall seconds** replace the 18,900-second launch reservation. The [sealed settlement](nfsp-delayed-R4-tabular-settlement.json) binds the original plan, run/result/audit hashes and releases the unused 9,649.646 seconds. Sixteen recorded tabular control/profile leaf trials now total **19,409.090 seconds (5.39141 hours)**, leaving **23,790.910 seconds** unallocated within the twelve-hour experimental-control ceiling. Tests, audits and bootstrap remain additional overhead. No remaining cohort or new target trial is selected by this settlement; the already funded neural R4 case continues separately.

The candidate-independent multigroup probe feasibility run completed 9,382 proposals in **180.021 CPU process-wall seconds** under a 180-second between-attempt ceiling. Its eleven prototype classes are preliminary construction evidence, with source/script and original artifacts retained in [probe-diversity-feasibility](../verification/probe-diversity-feasibility/README.md). The small production-constructor check added 3.230 seconds for its fixed 200-attempt prefix. Actual trace recertification, tests and independent audits are additional overhead not comprehensively summed here. These costs belong to probe construction, not the tabular-control ceiling or accelerator campaign pricing.


## Frozen multigroup construction reservation

The [single multigroup allocation](collision-probe-diverse-v1-allocation.json) reserves at most 1,800 seconds of new construction and 3,600 seconds of independent proof/source-seed verification under the design's eight-hour CPU probe stage. These are separately declared ceilings, **not completed costs**. Its source is isolated in a clean frozen worktree; the actual run/limits were checked against the allocation and original OS PID 2384063 was observed live. Prior construction measurements and additional audit/test/development overhead remain separately recorded. A partial catalog cannot pass coverage, and this reservation funds no training or confirmation inference campaign.


## Terminal multigroup construction

The original frozen construction has stopped at its unchanged cap: **1,800.007 CPU process-wall seconds**, 150,507 proposals and 24 distinct complete-game families. The roughly 0.007-second overrun is a bounded final attempt. This settles the construction portion as measured consumption; it does not select an extension or replacement. The [full terminal partial proofs](../verification/probe-diversity-v1-construction/README.md) remain below the required 32-family coverage.

The separately reserved 3,600-second exact-proof/source-seed audit (original PID 2393525) is now terminal and settled below. Its regeneration repeated the original proposal prefix solely for verification. Earlier construction/feasibility measurements and additional test/development overhead retain their existing scope; this ledger remains a partial cost record.


## Zero-record cohort CLI integration

Clean `7ab41a4` created three full-shape record-zero initialization/resume archives, registered them through the real CLI and verified read-only pilot/screening reports in **8.319 CPU process-wall seconds**. It collected **zero training records and zero scored games**. The [retained source-bound evidence](../verification/cohort-evidence-cli-diagnostic/README.md) reports missing-data/incomplete outcomes and no formal grants. Driver initial import/manifest setup, tests and development remain additional overhead. This metadata/report integration cost does not enter H_P/H_D/H_N or select learning work.


## Independent multigroup verification settled

The original audit is now complete and independently reproduces all 150,507 proposals and 24 exact families. Its measured interval is **1,794.814 CPU process-wall seconds**, including 2.587 seconds of proof reading and 1,792.223 seconds of regeneration. Together with the original 1,800.007-second construction interval, these two measurements total **3,594.821 seconds (0.99856 hours)**. The [settlement](collision-probe-diverse-v1-settlement.json) releases the unused 1,805.186 seconds of the verification reservation and selects no replacement construction or extension.

Initial setup/final reporting, prior construction/feasibility work, tests and development remain separately scoped; this is not complete project consumption. The [verified partial catalog](../verification/probe-diversity-v1-audit/README.md) remains below the required 32 families and yields no trained-policy/scientific outcome.


## Stationary supplemental-opponent diagnostic

The source-pinned forensic fixture uses 44 own-side root/suffix records and two demonstration gradient steps with exact oracle initialization. Its measured collector/target/loss interval is **0.680 CPU process-wall seconds**; initial import/manifest/setup and development overhead are additional. It is a bounded implementation diagnostic, not another full tabular/neural selection trial, an A2 pass or a large corrective allocation. The [confirmed passive-root bias](../verification/nfsp-forced-opponent-bias.md) requires an implementation correction and fresh matched pricing before further target selection.


## Complete version-two probe allocation settled

The separately selected two-point constructor completed 32 distinct families after 33,066 proposals in **417.546 measured CPU process-wall seconds**. Its full exact/source-seed audit took **421.945 seconds**, including 3.646 seconds of proof reading and 418.257 seconds of original-prefix regeneration. The combined measured intervals total **839.491 seconds (0.23319 hours)**; the [settlement](collision-probe-multicell-v2-settlement.json) releases unused construction/verification reservations and allocates no new training. Startup/final reporting, earlier construction work, tests and development remain separately scoped; this is not complete project consumption.

The [frozen complete benchmark](collision-benchmark-v2-freeze.json) resolves structural 32-family coverage. Teacher fixtures and exact proof construction do not establish learned behavior, controlled performance or gray specialization. The confirmed supplemental-opponent implementation defect must be corrected and repriced before further learner selection.


## Original R4 paired screen terminal

The matching original R4 neural case completed all 15M records in **9,611.440 CPU process-wall seconds**, with its frozen-source independent audit verified. The paired tabular/neural measured trial intervals total **18,861.794 seconds (5.23939 process-wall hours)**, separately sequential on CPU. Driver/bootstrap and independent audits are additional overhead. The tabular-control settlement remains unchanged; the neural case does not enter the twelve-hour tabular ceiling.

The [full neural receipt](../verification/nfsp-delayed-R4-neural.md) separates its passing numerical-average band from failed supported credit and the confirmed supplemental-opponent defect. No target/cohort/extension or corrected full training allocation follows automatically from these results. Old-source profiles cannot fund changed eligibility/cadence code.


## Corrected R1 profiles settled; one corrective pair selected

Four corrected source-bound 65,536-record profiles and all independent audits are complete. Measured case intervals including setup total **132.863 CPU seconds**, with a driver interval of 133.379 seconds. The two tabular leaves contribute 65.973 seconds, bringing eighteen recorded experimental control/profile leaves to **19,475.063 seconds (5.40974 hours)**. Audit/startup/test/development overhead retain their separate scope. The [profile settlement](nfsp-eligibility-R1-profile-settlement.json) releases unused reservation and retains the partial-buffer/timing-only limits.

The [separate corrective allocation](nfsp-eligibility-correction-R1-allocation.json) funds only a fresh seed-0 R1 tabular/neural pair, 15M aggregate records each, with 15,300 CPU-process seconds per case and margins at least two over the complete-source forecasts. Reserving the single tabular case projects **34,775.063 seconds (9.65974 hours)** within the twelve-hour experimental-control ceiling. No other control cohort or target is funded. The 30M learning records plus 262,144 profile records remain below the documented 45M corrective reserve. CPU-loop caps are not accelerator hours; no 9x9/full-study price or approval follows.


## Frozen-policy probe inference CLI diagnostic

The clean `f0751b0` zero-record, full-shape CPU diagnostic completed in **34.724 measured driver seconds**, including 1.318 seconds of initial network/archive preparation. Its persistent setup/inference/reproduction clock charges **23.148 seconds within that driver total**, against the original 600-second inference ceiling. A separately measured read-only raw-data/checkpoint/initial-parameter audit adds **0.243 seconds**. These are overlapping measurement scopes where stated; do not add the component clock to its driver total. Initial imports, failed preflight, regression/development and archiving overhead are additional.

The [retained evidence](../verification/probe-policy-cli-diagnostic/README.md) contains 384 actual zero-record policy marginals, 384 separately reloaded forward reproductions and read-only reporting. All three untrained policies grade 0/32 families. No learning records, accelerator or new experimental-control allocation, trained behavior or scientific conclusion follows from this diagnostic. The independent bounded-task auditor’s small development fixtures/tests are verification overhead, not formal A2 cohorts or an additional corrected target selection.


## Corrected R1 table settled; original neural case continues

The fresh corrected seed-0 table completed all 15M records in **7,431.799159 CPU process-wall seconds**, with its independent read-only audit verified and original inputs unchanged. The [settlement](nfsp-eligibility-R1-tabular-settlement.json) replaces the 15,300-second reservation, releasing 7,868.200841 seconds. Nineteen recorded experimental control/profile leaves now total **26,906.862258 seconds (7.474128 hours)**, leaving **16,293.137742 seconds (4.525872 hours) unallocated** within the twelve-hour ceiling.

The independent audit driver adds **32.506782 measured seconds** under its separate 600-second watchdog; setup/import/test/development overhead remains additional. The table misses supported credit and deployment readiness while greedy gaps are zero. The already allocated fresh neural case continues under original driver PID 2444001; no new cohort, reuse, target, extension or board work follows from this settlement.


## Matrix audit v2 historical verification settled

One predeclared read-only recheck of the original full 15M-record asymmetric tabular case completed in **8.587573 measured CPU driver seconds** under its 600/660-second watchdogs. Its [receipt and settlement](matrix-audit-v2-historical-recheck-settlement.json) reproduce the old numerical grade, retain original input checksums and keep one unresolved historical greedy label explicit. This is verification overhead, not another control trial, new learner source-reuse approval, A2 cohort or accelerator/campaign price. No original training handle was restarted.
