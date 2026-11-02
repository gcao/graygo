# Explicit fixed-shape paired evaluation batching

The selected R4 cost audit identifies scalar frozen-policy evaluation as a major component. The new `evaluation_batch.py` executes multiple complete games together, grouping frozen neural work while each game retains its original private actor RNG, opening, color assignment, episode-mixture component, resolver, score and trajectory hash. It does not change the learner or its clocks, train a policy, issue a gate receipt by itself, or select a campaign. Existing scalar calls remain available.

## Numerical policy identity

`FrozenNeuralPolicy(..., inference_batch_size=B)` explicitly selects a padded forward shape, with `1 <= B <= 64`. Shapes above one are restricted to the standard Q, policy and policy/value models, whose GroupNorm rows are independent. Batch shape can change floating-point reductions and exact Q ties, so it is part of the frozen policy identity together with the inference convention and device. Selecting B=32 is not a claim of bit identity with the older B=1 policy; old results are never relabeled.

Fixed shape alone proved insufficient: a CPU BLAS row microkernel could round Q ties differently at different row positions. Each encoded public observation is therefore assigned a fixed row by SHA-256 of its spatial/global/legal bytes. Lane collisions use separate forwards. Unused rows have zero spatial/global inputs and a pass-only mask; they are padding, never sampled or counted as games. Identical observations can share immutable probability vectors, but every actor draws its own RNG exactly once. A single-game/probe call uses the same fixed shape and row as a grouped call. Tests compare exact marginal/action/trajectory results across grouping, reordering and resume, including the full 64-channel/eight-block 9x9 CUDA network. No approximate tie rule or relaxed game gate is introduced.

Only public observations enter the grouped model. All private commitments in a wave are fixed before any board is resolved. Episode mixtures choose a component once using the existing private seed split. Candidate/incumbent games keep their common opponent seeds and both color games retain the same complete opening. Repeated trajectories are retained.

## Persistence and budgets

`EvaluationPlan.pair_batch_size` selects one to sixteen concurrent pairs. The immutable contract records the execution version, maximum physical games per group and budget boundary. Stop/time checks are between complete groups; a final group and IO can overrun, as explicitly recorded. With a matched incumbent a sixteen-pair group contains at most 64 physical games. Each complete paired unit is committed in original index order in its own SQLite transaction. A failed group yields no partial game or half-pair. A crash between unit commits preserves a contiguous complete-unit prefix; the original seeds and fixed numerical row convention reproduce the remaining work on resume.

Scientific batching operates within the existing atomic **48-game** matched block, keeping its cumulative clock, block validation and all contexts/arms/seeds/rules. It does not change endpoint counts, confidence coverage or independent units. Read-only evidence validates the recorded execution selection. Source/model/runtime/selection changes reject resume.

## Cohort and archive bindings

Before training, `DevelopmentCohort.register` can preregister the inference shape/device and pair count for both pilot and screening. The saved protocols, milestone plans and receipt readers enforce those settings. The independent score reader verifies the declared device's inference identity from archived weights on CPU; it does not allocate a GPU or sample games. The fixed seeds, windows, opening and strength bars remain unchanged. Registration and numerical receipts do not grant A2/resource/training-scope authority.

Scientific confirmation provenance reloads the selected inference convention from each verified public archive. Probe collection can select the same batch size, and independent CLI reproduction recovers each seed's size from the saved policy identity, along with the original runtime. This keeps deployed-game and solved-probe marginals on the same selected numerical policy.

## Commands

Ordinary evaluation keeps its existing required confidence and archive arguments, with these explicit options:

```sh
.venv/bin/graygo evaluate ... --device cuda --inference-batch-size 32 --pair-batch-size 16
.venv/bin/graygo cohort-register ... --device cuda --inference-batch-size 32 --pair-batch-size 16
.venv/bin/graygo science-evaluate ... --device cuda --inference-batch-size 32 --pair-batching
.venv/bin/graygo probe-evaluate ... --device cuda --inference-batch-size 32
```

Defaults retain B=1 and scalar pair collection. A supplied batch shape is immutable on resume. Full confirmation/archive and scientific controls still apply; these options do not bypass them.

## Verification and remaining pricing

The initial targeted evaluation/evidence group passes 29 tests. The complete suite passes **403 tests in 276.217 seconds**, with unchanged package source `f4c8b427d823d204df6bd46371b38b35bcb776d783ca04addd6831f84e666cbd`. Original verification driver/test PIDs **2571319 / 2571343** are terminal; the retained [full log](tests.txt) and [job binding](job.json) preserve that result. A separate committed-source CUDA CLI check is also complete: original corrected driver PID **2580561**, 12.896844 seconds, eight physical games/four paired units, source `a05655e`. It loads an existing full-network **record-zero** public average archive, preserves the first look exactly on resume, and rejects B=16 after a B=32 declaration without changing any evidence/input file. The [original job](cli-verified/job.json), [clean frozen-source report](cli-verified/resumed-look.json) and committed SQLite evidence are retained. This is CLI/persistence evidence, not strength or runtime pricing. The original setup attempt PID **2580305** used system Python and failed before Torch or any games; its [terminal failure record](cli-failed-system-python/job.json) is preserved. That setup failure was not an interrupted learning run or a reason to restart another process. Tests demonstrate implementation correctness at their stated cases, not convergence, trained strength or a full campaign price.

Forward-call reduction is not a throughput guarantee: hash-lane collisions, variable game lengths, tail padding, CPU resolver/heuristic work and policy diversity all affect actual cost. The old source-bound R4 prices and their 24/16-hour component overruns remain authoritative. A separately recorded selected-source complete profile and independent accounting review are required before replacing those prices or selecting a feasible training scope. No lower price is assumed from this implementation.
