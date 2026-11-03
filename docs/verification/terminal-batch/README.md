# Exact batched terminal replay collection

Source `2aa4f08` / package `d2743ee...` adds `ObservationSpec.pack_batch` and `ReplayBuffer.add_terminal_batch`. The matrix trainer defaults to the new path and retains the literal old collector as `scalar-reference`. No learning settings, labels, action/mode draws, update clocks, losses or numerical gates change. Existing scalar add and linked/nonstandard-storage paths remain intact.

Matrix actions/rewards/versions are already computed before writes, and learning occurs only after the completed chronological chunk. The optimized path validates arrays once, bit-packs observations by row and writes each retained ring slot once. Larger-than-capacity batches retain the same final slots/ordinals and historical/live Q counts. Reservoir offers and Algorithm R retain their original order/private stream; their callbacks do not depend on replay insertion timing. Sampling sees the identical completed buffer. Active stream tails and specialized schemas fall back to scalar add to preserve exact linkage. Collector mode is bound into manifest/private state and cannot change on resume.

The [original actual-learner comparison](comparison/job.json) completes **12 exact pairs / 24 fresh diagnostic runs / 2,124,288 own-side records**. Both games, both learners and closed/post-opening acting use real production losses/optimizers. Separate small-capacity post-opening stress wraps both replay and reservoir. Parameter/target/average states, full optimizer moments, private/global RNGs, replay/reservoir arrays, counters, scheduled public archives, numerical evaluation/loss fields and episode bytes match exactly. Only explicit source/environment identifiers, collector mode and timing fields are excluded. Mode labels and source identity are validated before comparison; changed learning values cannot be normalized away.

Finite observed production-layout ratios (scalar / batch):

| Case | Collection speedup, pre / post | Whole loop speedup, pre / post |
|---|---:|---:|
| asymmetric / tabular | 5.11 / 4.00 | 3.47 / 2.92 |
| asymmetric / neural | 5.00 / 4.21 | 3.16 / 2.56 |
| rps / tabular | 5.04 / 3.86 | 3.38 / 2.83 |
| rps / neural | 5.01 / 4.19 | 3.17 / 2.56 |

Across the fixed workloads, collection improves **3.86–5.11x** and loops **2.56–3.47x**. These are actual finite workload timings, not a universal speedup or a full 15M/board-study forecast. The production-layout runs have 200k capacity but partial population; separate 4,096-capacity wrap stress is not relabeled into a production buffer price. Old selected board-runtime source/prices are unchanged and the new source is not automatically admitted to the original A2 plan.

All **24 independent diagnostic audits pass** ([job](independent-audit/job.json)), checking unchanged source/case files, real update/target/average work, models, episode/private draws, replay, exact table publication bounds and original-game score reconstruction. Original root is `runs/terminal-batch-comparison-7d92069/measurement`; heavy tensors/replay/episode logs remain there. [Retention inventory](retention.json) binds copied reports and seals. The older producer's final intermediate progress file remains historical; original terminal job/report status is authoritative. The current harness now writes terminal progress explicitly.

The complete frozen-package suite passes **493 tests in 326.145 seconds** ([log](full-suite/tests.txt)). The first complete run had one pre-existing test issue: assigning utility `1.0` made no change when that row already scored `1.0`, so its expected checksum error could not occur. The [failed run](failed-full-suite/tests.txt) is preserved. The test now tries guaranteed different scores; nine promotion/calibration tests pass and production promotion checks are unchanged. Twenty-eight targeted storage/trajectory/resume tests and two actual comparison-harness tests pass as well. The literal scalar collector's source fingerprint remains preserved.

[Declared resource use](resource-use.json) preserves the completed first screens and all prior leaves. Profiling/control setup residuals remain reserved. Control commitment is **40,099.652475 / 43,200 seconds**; **3,100.347525** remain unallocated and select no new experiment. Known physical GPU allocation becomes **0.728515071444 device-hours**, counting both whole assigned suite intervals once, including the failed run. Independent audit parent time is not added to its child intervals; other unlisted tests/prototypes/history keep project consumption incomplete. CPU terminal timings price no C1/C4 or 9x9 route.

This completes the implementation/performance check. Same-protocol source admission and cohort aggregation are still required before new optimized A2 cases can join the six existing original cases. Thirty cohort slots, control affordability, route/project resource selection, production entry and scientific stages remain outstanding; no new formal case or global A2 pass is claimed.
