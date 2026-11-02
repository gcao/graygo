# Exact board milestone and opening journal

## Component and authority boundary

The journal connects the real batched board runtime to permanent public model archives, fixed opening decisions and complete resume bundles. It does **not** issue A2/resource approval or advance training itself. The production entry/scope/cohort executor must hold its exclusive execution lock, verify the global gates and call the journal before every collection chunk. No formal pilot or study is selected by creating this journal.

Run manifests freeze the runtime configuration, declared scope/record horizon, exact archive points, independent evaluation seed and training-start identity. Public `models.pt` archives use the existing verified loader/origin schema. Formal declarations check the full board architecture/cadence, primary schedule/optimizer settings, development versus confirmation seed sets and correct start distributions. Development uses official empty base-rule starts. Confirmation permits both base/remove rules and the declared empty/mixed training curriculum. Diagnostic declarations are bounded at 65,536 records and cannot issue formal decisions.

## Exact fresh-record boundaries

A public archive is written only when the actual runtime clock equals a declared point. It contains Q/average parameters, config, counters, run/plan/source and native-decoder identity; it never copies replay into the permanent weights archive. Idempotent capture verifies every current tensor against an existing bundle. Current weights cannot be relabeled as an earlier point.

`before_advance` rejects an unarchived current milestone, crossing the next archive/opening boundary, advancing beyond the planned cap, or proceeding past an undecided/no-go opening. The runtime separately enforces paired chunks, optimizer cadence and hand-off boundaries. Thus 4.75M/5M archives do not drift to a later optimizer boundary. The fixed 40M epsilon clock, old frozen episode policies and replay history stay intact.

## Fixed opening and real evidence

At the original 5M development boundary, archive and checkpoint the unchanged current Q state with an empty reservoir. The formal decision loader reconstructs the declared greedy-Q milestone objects and exact evaluation plans from actual scored directories. It checks registered run origins, baseline identities, rules/official empty openings, representation, independent derived seeds, confidence family and largest completed look. It then applies `PilotProtocol`'s same-snapshot two-opponent/two-of-three rule. Missing reads remain pending. A final pass or no-go is immutable; exhausted/inconclusive evaluation preserves a closed-reservoir no-go. The journal cannot select an older favorable Q checkpoint or move opening.

Confirmation follows the fixed opening schedule. Its journal refuses the development performance-filter method. Later post-opening reads do not clear/reset label history. The whole three-seed 8M/20M cohort decisions and campaign allocations remain responsibilities of the outer controller.

## Atomic checkpoints and crash intervals

The runtime save API accepts a JSON controller payload. `controller.json` commits in the same sealed bundle as model/target/optimizer, full buffers, environment/frozen-actor state and RNGs, with its own canonical hash in metadata. The journal validates the run/plan/clock/archive prefix before resuming.

An opening decision may arrive after the pending 5M checkpoint, while the learner clock remains exactly 5M. It is a separately immutable control artifact bound to that run, plan and exact permanent model bundle. On recovery, the journal can apply it only at the original opening, only after reproducing the same current parameters, and only with the original closed reservoir. A post-opening checkpoint without its recorded decision is rejected. Same-clock checkpoints are not overwritten; only this verified control overlay can differ.

Permanent archives created after an older resume point remain retained. Replay must reproduce their tensors at the correct clock before adopting them. A mismatched replay fails instead of importing future parameters or silently skipping training. Public weight loading now uses a forked Torch RNG context, so constructing a model just to load its public weights does not consume the training RNG. CPU/CUDA states are checked explicitly.

Finalization requires the declared endpoint or a sealed opening no-go. Active/budget-limited work retains its resume discipline. Completed finalization keeps one verified heavy bundle and all permanent milestone archives. The outer controller still owns cumulative time accounting and the project resource/scope decision.

## Verification and scope

Tests exercise exact archive clocks/origins, RNG-neutral public loading, forbidden boundary crossing, zero early label insertion, immutable no-go, corruption rejection, changed resume plans, mismatch with a future permanent archive, and exact CPU/CUDA continuation across the post-decision/pre-checkpoint crash interval. They also check retention and refusal to finalize an active partial run. The combined suite passes **227 tests in 85.483 seconds**, including the exact CPU/CUDA opening-overlay cases. Bounded full-model diagnostic evidence is recorded separately after collection.

The diagnostic uses a reduced opening and record cap to test lifecycle mechanics. It is not a 5M readiness outcome or an A2/profiled resource approval. Formal all-seed evidence, the production entry/cohort executor, selected-learner campaign pricing, scientific training and complete independent probe coverage remain required by the full design.


## Completed full-model diagnostic and independent audit

Clean revision `546d2e2`, package SHA `18bc62c487c4d90dcad1814b15705a89c3ca743ef8c717f5ffa29119cbad15f6`, completed two full 64-channel/eight-block CUDA trajectories of 16,384 fresh records each. Total elapsed process time was **87.245 seconds** under a frozen 300-second cap. The [plan](board-milestone-diagnostic/plan.json), [report](board-milestone-diagnostic/report.json), [independent audit](board-milestone-diagnostic/audit.json) and [retained-artifact index](board-milestone-diagnostic/README.md) bind the source, configuration and sealed evidence.

Each trajectory performed 100 Q updates / 51,200 Q targets and ten averaging updates / 5,120 supervised targets, with one fresh-record target refresh. The reservoir was empty at the reduced 8,192-record opening; its final 903 labels all have insertion clocks at or after that opening. Both trajectories retain all five exact public archives and one final heavy resume bundle. The independent reader verifies the actual seals and arrays, including tensor dtypes, optimizer/target/collector state, private CPU/CUDA RNGs and every allocated replay/reservoir array. All match exactly after recovering the separately written opening control. Public archive loading preserves the training RNG.

This check uses **reference Python decoding**, Q reuse four and reduced diagnostic opening/hand-off clocks, with untrained game lengths. It cannot isolate reuse timing against the earlier R1/native-decoder profile, establish H_P/H_D/H_N, qualify the formal all-seed pilot or supply A2/resource authority. No production pilot or confirmation training was launched.
