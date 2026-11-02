# Original-cohort archive and scored-evidence verification

## Immutable original run registry

`DevelopmentCohort` registers the original development seeds 100/101/102 together at record zero. Every run must declare the same full primary 9x9/200-round architecture, optimizer, cadence, start distribution, fixed 5M/7M opening/hand-off, 20M horizon and archive windows; only the training seed differs. It requires current frozen source, original numerical dependencies/Python, no post-initialization archives or opening decision, sealed public/resume bundles labeled at record zero and three distinct run fingerprints.

The reader reproduces each Q/average initialization from its declared CPU RNG seed and exact model-construction order. Distinct arbitrary or resealed pretrained weights are insufficient. Initial weights, original run/config and archive seals remain bound by the registry checksum and are rechecked on load. Registration is immutable; it cannot retrospectively choose three favorable runs or replace a failed seed. It grants no training or evaluation allocation.

## Read actual archived policies and scored rows

The registry fixes both confidence protocols and the evaluation seed namespace before data collection. `plan_for` derives each original endpoint, private evaluation seed, confidence family, public model path and expected evidence folder. Only Q is a pre-opening representation; post-opening reads include Q and mandatory average diagnostics; screening uses the average at 20M.

| Stage | Required scored endpoints |
|---|---:|
| Pre-opening Q: 3 seeds x 3 snapshots x 2 opponents | 18 |
| Post-opening Q: 3 seeds x 3 snapshots x 2 opponents | 18 |
| Post-opening average diagnostics | 18 |
| Pilot total | 54 |
| Screening average: 3 seeds x 2 opponents, in addition to the pilot evidence | 6 |

`FrozenEvaluationEvidence` opens SQLite with an explicit read-only URI and a consistent read transaction. It validates unit prefixes/checksums, complete opening hashes, score totals/signs, paired aggregates, possible terminal lengths and the private episode-mixture draws. It rejects a missing database rather than creating one. Both current pairing/rule versions are required; historical nonempty version-1 pairing cannot be silently reinterpreted.

A milestone read reconstructs the exact frozen Q/average parameters from its permanent sealed archive, checks source/run/journal/native-decoder and fresh/update/refresh accounting, and compares the complete expected evaluation contract. The two baselines use the same archived model, the actual versioned random/tactical identities, official empty starts, fixed representation and independent derived seeds. Current/future weights cannot be relabeled as another point, and edited summary estimates are not inputs.

Bounds are recomputed from scored color pairs. The largest completed registered look (500 or 2,000 games per matchup) is selected automatically; an appended later look invalidates an older snapshot. Repeated trajectories remain observations. A saved historical prefix can still be estimated explicitly by the low-level evidence reader, while the live cohort gate always uses its largest completed look.

## Cohort rules and critical read interval

Pilot reports apply the original same-snapshot two-opponent, two-of-three rule in each fixed window, require all three original seeds and retain all eighteen post-opening average diagnostics. Screening first rechecks the complete pilot evidence, then requires every original 20M average to have random lower utility strictly above zero and tactical lower utility at least +0.10. No seed, checkpoint window, representation or endpoint is substituted after observing data.

A snapshot holds shared OS locks on every present evaluation input across the entire read interval. Collectors require exclusive locks and cannot append a later look during that interval. Callers can keep `snapshot` open while recording a controller decision. The public report returns an as-of snapshot, never a formal grant. Missing/partial scored inputs remain incomplete, and malformed or mismatched inputs fail verification.

The fixed-opening journal now uses this same read-only milestone path. It keeps all six present pre-window inputs shared-locked through the atomic immutable 5M decision write. It accepts only the original pre-window keys, preserves the closed reservoir while evidence is pending, and cannot use post-opening reads to justify an earlier opening. It no longer instantiates a read/write evaluation collector to analyze scored evidence.

## RNG and input preservation

Public archive loading verifies bundles statically and creates no checkpoint-management directories. Temporary model construction is forced onto CPU inside a forked CPU RNG context, even if a caller's default device is CUDA. It preserves live CPU/CUDA training RNGs. The cohort reader likewise reproduces initial weights without reseeding the live CUDA generators.

Registration writes its one immutable registry; reporting writes no input artifact. The CLI rejects an output path inside the registry or any original training run, so a report cannot overwrite `cohort.json`, `run.json`, model/checkpoint data or scored inputs. Requested report outputs belong outside those frozen inputs.

## Authority and verification limits

These are conditional numerical gates over sealed original archives and scored rows. The component does not independently reconstruct the complete training trajectory or every evaluation move from trajectory digests. Full optimizer/buffer/private-stream reset verification belongs to the entry/execution controller; this reader independently reproduces the initial model parameters. It also does not independently verify evaluation numerical-runtime context, A2 eligibility, the project resource/scope decision, actual opening/control history or approved cohort advancement. The production entry/cohort executor must bind those prerequisites and cumulative collection/evaluation budgets before formal learning. A numerical pass here always retains `formal_scope_approved=false` and `a2_complete=false`; it cannot authorize screening, confirmation or protected promotion by itself.

The numerical confidence statements condition on the fixed registered cohort. Game counts do not create a training-seed population interval. No real development pilot or confirmation campaign is launched by registering or reporting evidence.

## Verification result and commands

The combined suite passes **272 tests in 161.982 seconds**. Eight low-level reader tests use actual small scored games and verify read-only files/directories, exclusive/shared lock behavior, matched incumbents, private mixture draws, remaining-round limits, mutated snapshots, raw-score tampering and later completed looks. Eleven cohort tests cover seeded fresh originals, immutable registration, exact archive/score/protocol bindings, all-seed/average requirements, input preservation, lock retention through opening commit, resealed changed initial weights, report output protection and CPU/CUDA RNG preservation under a default CUDA context.

Positive cohort gate tests use explicitly synthetic future-clock archives and scored headers to exercise protocol/verification mechanics; they are not learned-strength or completed-training evidence. A real CLI integration check uses only full-shape zero-record initialization archives and expects missing-data reports; its terminal artifacts are recorded separately after collection.

```sh
# Only register original initialized runs; these commands collect no records/games.
.venv/bin/graygo cohort-register --run-dir runs/config/cohort --name config --training-run 100=runs/config/seed100 --training-run 101=runs/config/seed101 --training-run 102=runs/config/seed102
.venv/bin/graygo cohort-report --run-dir runs/config/cohort --stage pilot --output reports/config-pilot.json
.venv/bin/graygo cohort-report --run-dir runs/config/cohort --stage screening --output reports/config-screening.json
```

Scored folders are fixed under `cohort/evaluations/{seed}/{record:012d}/{representation}/{opponent}`. `plan_for` returns the exact evaluation plan, confidence family and original archive/evidence paths for the eventual approved collection controller. The registry is source/environment bound; use its frozen code and dependencies to re-read it.
