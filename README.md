# Gray Go

Simultaneous Go on a 9×9 torus. Both players secretly choose a placement or pass; collisions create neutral gray stones. The game uses capture-before-suicide resolution, player-specific restrictions, two-round ko history, and exact fractional area scoring.

The authoritative [design](docs/design.md) includes rules, the neural-network architecture, training gates, and review history. [Implementation status](docs/implementation.md) tracks the full project against it.

## Setup

Python 3.12 is the tested interpreter. From the repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train]'
.venv/bin/python -m unittest discover -s tests -v
```

The core rules/CLI need NumPy. The `train` extra adds Torch and SciPy for networks and certified numerical game solving. The tested Linux/CUDA environment is recorded in `requirements-linux.lock`; it is specific to the `home-ai` target.

## Commands

```sh
# Human black versus a tactical white opponent. Coordinates are 1-based.
.venv/bin/graygo play --seed 42

# Baseline games; each agent commits from the same immutable public state.
.venv/bin/graygo simulate --games 10 --seed 42 --output artifacts/games.json

# Fixed reference-engine workload and full-network synthetic timing.
.venv/bin/graygo profile --rounds 100000 --seed 42
.venv/bin/graygo profile-network --device cuda --batch 16 --batches 10

# Fully enumerate a bounded research game. Other sizes require a limit.
.venv/bin/graygo solve --size 3 --limit 1
```

`simulate`, `profile`, and `solve` produce JSON with rule/schema versions, configuration, source fingerprint, seed, dependencies, hardware, and results. `--output` writes the report atomically with the disk-reserve guard. A solver budget exhaustion returns an error rather than a heuristic solution.

The optional `audit-native --library PATH` command compares reference-engine transitions and the full observation against the review prototype ABI. That external library is not a production dependency.

## Current implementation

The repository contains the immutable reference engine, full-state encoding, random/tactical baselines, policy/value/Q networks, finite-game solvers, checkpoint storage and recovery, and CLI commands. The completed PPO matrix preflight selected NFSP; its independently audited evidence is in [the preflight report](docs/verification/ppo-preflight.md). The shared NFSP update/collection backend, compact buffers and solved delayed/branching controls are implemented. The resumable neural/tabular NFSP matrix driver is available; multiround solved-task training is also integrated; formal convergence gates, production 9×9 stages and scientific comparisons remain in progress; synthetic updates are not trained-game results.

## Learning preflight

```sh
# Individual matrix run, including optimizer/RNG/opponent/archive resume state.
.venv/bin/graygo train-matrix --task asymmetric --entropy taper --run-dir runs/asymmetric-seed0
# Repeat the same plan with --resume to continue an interrupted run.
.venv/bin/graygo train-matrix --task asymmetric --entropy taper --run-dir runs/asymmetric-seed0 --resume

# The formal 12-case route preflight: two tasks, two entropy protocols, three seeds.
.venv/bin/graygo preflight --run-dir runs/ppo-preflight --device cpu
```

The preflight first checks tabular exact-best-response fictitious-play reference results. This is separate from the later NFSP Q/replay/reservoir control. Each neural trial uses the shared PPO update code, 50/50 current/history opponents selected before commitments, and critic-only opponent buckets. In a one-step game, the stored two-role probability table is an exact frozen historical policy.

Preflight runs are resumable by repeating the command with the same plan. Finished trials are checksum-verified and reused. `summary.json` selects a route only after all 12 formal trials have complete evidence; shortened diagnostics report `pending`. A passing protocol must use one deployment representation consistently across both games and all seeds. SIGINT/SIGTERM requests stop at a rollout boundary and save state.

## NFSP verification

The one-step backend in `nfsp.py` is shared by compact task and board models. `nfsp_collection.py` provides frozen per-episode native self-play and independently sampled simultaneous actions. `tasks.py` supplies opaque delayed/branching controls and balanced value-only coverage; `public_game.py` computes exact fixed-policy/action values and responses. The resumable matrix trainer includes an exact reservoir-average tabular control and a matched executed-action-label control for `pure82`. Formal A2 learning evidence and the production 9×9 stages remain required.

```sh
.venv/bin/python -m unittest discover -s tests -v
# Partial storage benchmark; includes checksummed save/load, not full learner cost.
.venv/bin/graygo profile-buffers --buffer-dir artifacts/buffers --records 10000
```

See [NFSP backend verification](docs/verification/nfsp-backend.md) for the checks and their limits.

## NFSP matrix training

```sh
# Primary 15M-record recipe; 5M reservoir opening, 5M–7M hand-off.
.venv/bin/graygo train-nfsp --task asymmetric --learner tabular --run-dir runs/nfsp-asym-table-seed0
.venv/bin/graygo train-nfsp --task asymmetric --learner neural --run-dir runs/nfsp-asym-neural-seed0
# Continue the identical plan after interruption.
.venv/bin/graygo train-nfsp --task asymmetric --learner neural --run-dir runs/nfsp-asym-neural-seed0 --resume
```

`--records`, `--reservoir-open` and `--handoff-end` can produce diagnostic profiles; such changed recipes cannot pass the formal-protocol flag. `--label-rule executed` is restricted to `pure82`. The exact tabular average is published at supervised cadence, while Q uses the same optimizer/loss/replay path as the neural learner. Runs save permanent public weights, bounded resume bundles and a checkpointed binary per-episode audit. SIGINT/SIGTERM saves at the next balanced collection chunk.

## Multiround NFSP tasks

```sh
# A short profile uses the real multiround collector/update path and remains nonformal.
.venv/bin/graygo train-nfsp-task --task delayed --learner neural --purpose profile --records 50000 --time-cap-hours .25 --run-dir runs/delayed-profile
# Actual reachable mixed Gray Go root, complete tabular state identity.
.venv/bin/graygo train-nfsp-task --task gray-mixed --learner tabular --purpose profile --records 10000 --time-cap-hours .25 --run-dir runs/gray-mixed-profile
```

`delayed-forced` retains the forced-action indexing diagnostic; `delayed` uses 40 equivalent intermediate choices by default. `branching` gives later actions meaningful payoffs. Bounded neural Go tasks use the full board encoder. Resume with the identical command plus `--resume`; active episodes retain their original frozen parameters even if the learner has updated since episode start. Use `tools/profile_nfsp_tasks.py` from a pinned worktree for complete paired cost profiles before full A2 allocations. The budget artifact must match source, task, learner and Q reuse. Full A2 exit still requires the specified cohorts and all gates.

## Compact board storage

```sh
# Compare storage/restore/regeneration cost using legal game data.
.venv/bin/graygo profile-buffers --storage rule-state --buffer-dir artifacts/rule-buffers --records 20000
# Bounded neural training uses the same update path with regenerated features.
.venv/bin/graygo train-nfsp-task --task gray-mixed --learner neural --storage rule-state --purpose profile --records 10000 --time-cap-hours .25 --run-dir runs/compact-gray-profile
```

Rule-state packets retain the full public game state and player perspective. Masks/global planes are regenerated rather than duplicated in host arrays. Feature-storage snapshots remain supported. Timing budgets must match the storage choice; the reference Python decoder's cost is not yet a production throughput claim. See [storage verification](docs/verification/rule-state-storage.md).

## Native compact decoding

```sh
# C++17 compiler required; the JSON result identifies an immutable library path.
.venv/bin/graygo build-native --output-dir artifacts/native
# Select that exact path explicitly for compact-buffer profiling or board training.
.venv/bin/graygo profile-buffers --storage rule-state --native-library PATH_FROM_BUILD --buffer-dir artifacts/native-buffers --records 20000
```

`train-nfsp-task` also accepts `--native-library` with `--storage rule-state`. Snapshot metadata and budget checks bind the binary SHA-256 and ABI; keep that build available when resuming. Unsupported rules/ABIs or changed binaries fail explicitly. The independent Python decoder remains the reference. Native code is for batched feature generation and validation, with no game-rule or learning-target changes. See [verification](docs/verification/native-rule-decoder.md).

## Frozen paired evaluation

Create a `ConfidenceFamily` with the complete endpoint/look set before collection, save its `to_dict()` as JSON, then run:

```sh
.venv/bin/graygo evaluate --name baseline-read --subject-label baseline --endpoint utility --subject random --opponent tactical --confidence-plan confidence.json --run-dir runs/baseline-read --games 500 --time-cap-hours .5
# The same contract can continue to a predeclared later look.
.venv/bin/graygo evaluate --name baseline-read --subject-label baseline --endpoint utility --subject random --opponent tactical --confidence-plan confidence.json --run-dir runs/baseline-read --games 2000 --time-cap-hours 1 --resume
```

Neural subjects use `--subject neural --subject-archive PATH --representation average` (or `greedy_q` for the separate Q diagnostic). `--incumbent` adds matched improvement evidence; its endpoint type is fixed before the first game. Whole color pairs commit atomically. Repeated game hashes are reported rather than removed. The plan binds confidence method/family, models, source, openings and sample looks. Bounds do not constitute a promotion certificate: joint protected-pool calibration and the other design gates remain required.

`PilotProtocol`/`ScreeningProtocol` require registered development-run hashes and frozen baseline identities; serialized constants are checked on reload. The pilot requires both fixed Q windows and post-opening average diagnostics for all three development seeds. Confirmation seeds use the fixed opening and are never performance-filtered.

## Protected promotion calibration

`ProtectedPool` and `PromotionProtocol` freeze opponents, weights, rules, openings, all planned sample looks and numerical tolerances. Candidate pair plans must bind `decision_protocol_sha256` and the successful pre-candidate `calibration_sha256`; statistical bounds alone are insufficient to promote.

```sh
.venv/bin/graygo calibrate-promotion --protocol promotion-protocol.json --pilot random=runs/pilot-random --pilot tactical=runs/pilot-tactical --run-dir runs/gate-calibration --simulations 2000 --seed 7 --time-cap-hours .5
# Same immutable plan and cumulative budget after interruption:
.venv/bin/graygo calibrate-promotion --protocol promotion-protocol.json --pilot random=runs/pilot-random --pilot tactical=runs/pilot-tactical --run-dir runs/gate-calibration --simulations 2000 --seed 7 --time-cap-hours .5 --resume
```

The full joint simulator includes zero/+0.03/nontransitive/regression/floor and missing-readiness cases, with separately bounded simulation uncertainty. It preserves infeasible or underpowered outcomes as `not_calibrated`. The provided small baseline variance pilot is not a qualified promotion calibration. Actual promotion decisions still require the verified project-level A2 and 9×9-smoke bridge. See [procedure and limits](docs/verification/protected-promotion.md).

## Real-board collection and bounded learner profiling

`BoardNFSPRuntime` combines private batched commitments, complete public rule-state replay, independent Q/average networks, the fixed 40M epsilon clock, exact optimizer/refresh counters, and episode-frozen actors. A checkpoint preserves active environments, their old policy versions, both buffers, augmentation/private RNGs and optimizer state. CPU and CUDA continuation are tested, including the full 64-channel/eight-block CUDA model.

```sh
.venv/bin/graygo profile-board --run-dir runs/board-hardware-R1 --device cuda --envs 16 --records 16384 --q-reuse 1 --native-library PATH_FROM_BUILD --seed 17 --time-cap-hours .2
```

This command measures two fresh, bounded hardware phases: closed-reservoir collection and a diagnostic early opening to exercise supervised updates and average-policy acting. It retains the logical batch of 512, target refresh at 12,800 fresh records, paired geometric augmentation and strict native decoder. Each phase records game/gray/restriction statistics, Q/SL work, checkpoint write/restore, memory and small paired evaluation timings. Check the coverage flags: a short or interrupted profile may not exercise every component. The total record cap is 65,536 and the total time ceiling cannot exceed one hour. Inspect its original process and output before starting another measurement; the CLI does not silently resume or overwrite a profile.

The diagnostic early opening is explicitly different from the production 5M–7M hand-off. These untrained game lengths and acting proportions do not establish a full campaign price or training strength. The component is not the production development/confirmation controller: A2 readiness, fixed milestone evaluation, label-opening authorization, cohort decisions and resource allocation still need that integration. See [runtime verification](docs/verification/board-runtime.md).


## Frozen gray-exposure curriculum

```sh
.venv/bin/graygo generate-exposure --seed 8675309 --catalog artifacts/gray-exposure-v1.json
```

The versioned generator creates 64 training and 32 held-out 9x9 families without a candidate policy as input. Boards contain one to six gray stones in groups of at most three and zero to twelve balanced black/white pairs. Every group has liberties. Family identity removes rotations/reflections, torus translations and color exchange, so symmetry copies cannot leak between splits. Loading verifies the complete schema, family IDs and regeneration from the original seed; the command preserves an existing identical catalog and refuses to overwrite a different one.

`ExposureStarts(catalog)` supplies the prescribed 50% empty / 50% gray-start distribution with random torus/D4/color transforms and fresh 200-round metadata. Use `ExposureStarts(catalog, 'heldout', 'gray-only')` for the separate gray-start diagnostic. A given sampling seed produces the same board and family under either collision rule. The real collector records start/gray/family counts and binds the catalog/split in its checkpoint; recovery rejects a changed split.

This is the candidate-independent starting distribution component. The paired scientific controller must still match/record actual start consumption and fresh-record exposure across arms; different game lengths can consume different numbers of starts. These held-out families are not the separately certified collision-probe benchmark. See [curriculum verification](docs/verification/gray-exposure.md).


## Exact collision-probe certificates

```sh
.venv/bin/graygo build-probes --seed 473927 --run-dir runs/collision-probe-catalog-v1 --attempts 10000 --time-cap-hours .05
.venv/bin/graygo verify-probes --benchmark runs/collision-probe-catalog-v1/benchmark.json --regenerate --time-cap-hours .05
```

The certifier resolves every legal action pair at a public state with exactly one round remaining, proves the game value with exact rational primal/dual witnesses, and exhaustively bounds collision probability over all strategies whose individual gaps are at most 0.05. Partial enumeration or a restricted action set cannot qualify a case. Pass is excluded from collision probability. Certificates can be verified without calling the floating LP solver.

The first constructor provides 32 geometric regression pairs with 64 distinct current boards and full public roots. Every root is reached from empty by a legal 199-round trace under the default 9x9/200-round rules; paired histories differ only at the controlled stone. All pairs have a Gray-rule utility effect and the required collision-range separation. **They share one strategic motif, so the 32-independent-family requirement remains incomplete.** The coverage checker clusters complete base/remove game pairs under action relabeling, color exchange and pair reversal; repeated strategic instances cannot pass the family gate.

`assess_marginals` grades both roles and both roots for all three fixed seeds. Instance counts and independent-family counts are separate; all instances in a repeated family must pass, and official acceptance requires complete family coverage plus at least 24 passing families per seed. `frozen_marginals` uses public neural probabilities and private episode-mixture initial marginals. Each probe episode begins at its declared root; the setup trace establishes reachability. Continuing mixtures after an acting-policy prefix require posterior conditioning, and the default mixture API rejects an implicit reset.

The generator starts no training and yields `incomplete_family_coverage` for the current one-motif catalog. Qualified bounds, reachability and passing solver fixtures do not establish trained-policy strength or gray specialization. See [proof method and evidence](docs/verification/collision-probes.md).


## Matched scientific rule comparison

`science-evaluate` accepts a frozen JSON registry mapping every `curriculum/rule/seed` key (curricula `empty`/`exposure`, training rules `gray`/`remove`, seeds 200/201/202) to a permanent public weight archive:

```sh
.venv/bin/graygo science-evaluate --name gray-study --catalog artifacts/gray-exposure-v1.json --registry policy-registry.json --run-dir runs/gray-study --games 24000 --time-cap-hours 12 --device cuda
.venv/bin/graygo science-evaluate --name gray-study --catalog artifacts/gray-exposure-v1.json --registry policy-registry.json --run-dir runs/gray-study --games 96000 --time-cap-hours 12 --device cuda --resume
.venv/bin/graygo science-report --run-dir runs/gray-study --games 96000
```

The numeric ceiling above is an example, not an allocation. The campaign scope decision must fund scientific evaluation and probe inference together within their recorded ceiling. Confirmation inputs require twelve distinct verified average-policy archives at the 80M milestone, their original confirmation seeds/rules, and the exact declared training starts. Development/gate agents and relabeled baselines fail before data collection. The production controller must separately verify A2, training scope, matched actual exposure and probe coverage before a scientific claim.

One atomic block includes both curricula, official-empty and shared held-out-gray roots, all three seeds, both evaluation rules and both agent colors: **48 games**. Planned looks of 500/2,000 blocks are exactly 24,000/96,000 total games. The interaction uses matched within-block differences; simultaneous bounds cover all registered aggregate/seed diagnostics and looks. The primary exposure-controlled official-empty endpoint needs both lower bounds above +0.05 and positive point estimates in every seed. Empty-trained effects remain secondary, and held-out-gray effects remain diagnostic.

Plans, policy/runtime/catalog identities, scored blocks and a checksummed cumulative clock persist across resume. A failed partial block commits no subset. An unclean in-flight gap is conservatively charged, including offline time, so crashes cannot silently refresh the time budget. The read-only reporter never mutates evidence and refuses a hidden later completed look. Small baseline exercises require `--purpose diagnostic`, explicit small `--block-looks`, and `baseline:random`/`baseline:tactical` registry values. They cannot establish a confirmation result. See [verification and inference scope](docs/verification/scientific-contrast.md).


## Exact board milestones and opening journal

`BoardMilestoneJournal` permanently archives the current Q/average weights at exact fresh-record boundaries and binds its lifecycle state into atomic runtime checkpoints. Development declarations retain the 4.5M/4.75M/5M pre-window, 6M/7M/8M post-window and 20M horizon; confirmation declarations retain the fixed opening and 80M horizon under either rule. The journal records a scope declaration, while the project controller must separately verify A2, resources and cohort approval before starting any formal training.

Call `capture_if_due()` and `before_advance(count)` before collecting a chunk. At the fixed development opening, collection stops with a closed reservoir until `record_pilot_opening` reconstructs the real frozen milestone evaluations and records the original pre-window decision. A pass/no-go file is immutable; incomplete reads remain pending, and no later window can replace a final no-go. Confirmation label opening follows the fixed schedule and cannot be performance-filtered.

```sh
# Bounded diagnostic: full models, real rules, 512-target updates, CUDA and
# a deliberately reduced opening to exercise the lifecycle in little work.
.venv/bin/python tools/check_board_journal.py --run-dir runs/board-journal-check --device cuda --time-cap-seconds 300
```

The check compares uninterrupted and resumed 16,384-record trajectories, including a crash after the opening decision but before another full checkpoint. Public weight loading preserves training CPU/CUDA RNGs. All declared weights remain permanent; completed runs retain one verified heavy resume bundle. An active/time-capped run cannot be finalized as a completed endpoint. See [journal verification and remaining controller work](docs/verification/board-milestones.md).


## Locked board execution diagnostics

`run-board-diagnostic` executes the real board runtime under an exclusive process lock and a persistent resource clock. It stops exactly at public archives, optimizer/checkpoint boundaries and an undecided opening. Resume preserves the original source/config/ceiling and charges each invocation's setup; an unclean interval is conservatively charged. Disk checks cover full-capacity checkpoint headroom, and safe partial stops preserve recovery generations.

```sh
.venv/bin/graygo run-board-diagnostic --run-dir runs/board-execution --records 16384 --until-records 4096 --time-cap-hours .08333333333333333 --device cuda --q-reuse 4
.venv/bin/graygo run-board-diagnostic --run-dir runs/board-execution --records 16384 --time-cap-hours .08333333333333333 --device cuda --q-reuse 4 --resume --opening-decision pass
```

This command is bounded to 65,536 records and one cumulative hour, with a deliberately reduced diagnostic opening/hand-off. Its manual opening control is diagnostic only. The full formal entry/A2/resource/cohort controller remains required before development or confirmation training. See [execution verification, failure handling and scope](docs/verification/board-execution.md).


## Distinct multigroup collision probes

`build-probes --generator multigroup` searches candidate-independent removal/recoloring pairs with varied black/white/gray groups and permanent liberties. It retains one pair per complete base/remove game class, rejects geometric repeats and builds every accepted root from empty through 199 actual rounds. Exact 0.05 individual-gap collision ranges must remain separated by at least 0.10; all actions and pass remain in the solved game. Gray setup stones are produced by actual collisions.

```sh
.venv/bin/graygo build-probes --generator multigroup --seed 974214 --run-dir runs/collision-probe-diverse-v1 --attempts 200000 --time-cap-hours .5
```

Only a complete catalog of 32 distinct classes, balanced sixteen increase/sixteen decrease, can pass family coverage. Partial proofs and actual counts remain visible. The historical liberty generator stays available for its one-class geometric regression catalog. See [construction, legal setup, exact qualification and evidence scope](docs/verification/probe-diversity.md).


## Original development cohort evidence

`cohort-register` freezes the three original seeded initializations before training. `cohort-report` reads permanent milestone models and raw scored SQLite inputs, recomputes the largest completed confidence looks and enforces the fixed three-seed pilot/screening rules. It retains mandatory average diagnostics, rejects changed source/seeds/models/baselines/openings and takes shared input locks while reporting. It writes no frozen input artifact.

```sh
.venv/bin/graygo cohort-register --run-dir runs/config/cohort --name config --training-run 100=runs/config/seed100 --training-run 101=runs/config/seed101 --training-run 102=runs/config/seed102
.venv/bin/graygo cohort-report --run-dir runs/config/cohort --stage pilot --output reports/config-pilot.json
```

These commands collect no training records or evaluation games and issue no A2/resource/training authority. A numerical gate result remains conditional on the independently verified project prerequisites. See [archive/score verification, locking, protocol rules and remaining authority](docs/verification/cohort-evidence.md).


## Controlled changes at two points

The version-two probe constructor (`build-probes --generator multicell`) explores two occupied-cell removals/recolorings while preserving the exact certificates, legal 199-round setup, uniqueness and 32-family requirements. It retains all original holes and limits either root to five empty points, so pass and every legal placement still fit the six-action proof budget. Catalogs record both changed positions explicitly. The separately versioned one-cell generator and its audited partial catalog remain unchanged. See [method, compatibility and verified test scope](docs/verification/probe-multicell.md).


The complete [32-family version-two benchmark](docs/verification/probe-multicell-v2-construction/benchmark.json) is now frozen and independently source-seed reproduced. It contains 64 distinct reachable roots with exact full-action certificates and balanced collision directions. Local canonical copy: `artifacts/collision-benchmark-v2.json`. Structural coverage is complete; [audit scope](docs/verification/probe-multicell-v2-audit/README.md) remains separate from learned-policy and controlled-performance claims.


## Supplemental value eligibility

Supplemental coverage retains the forcing role's conditional root-action target and both normal suffixes, while excluding the other role's opponent-intervened root target. Every collected own-side record still counts toward the aggregate allocation, epsilon and 5M/7M schedule. Only value-eligible records drive Q reuse and target refresh; replay and losses enforce that mask. Complete-round checkpoints preserve both clocks, per-role eligibility and exact private-RNG continuation. See [the causal correction and evidence scope](docs/verification/nfsp-q-eligibility.md).

## Frozen collision benchmark policies

`probe-evaluate` collects both roots and roles for each of the 32 frozen families across all three confirmation seeds. Confirmation binds actual permanent average-policy archives to the frozen scientific registry; explicit diagnostics cannot pass confirmation. `probe-report` is read-only. `probe-audit` repeats the original frozen model forwards under the same cumulative inference ceiling. See [the evaluator contract and commands](docs/verification/probe-policy-evaluation.md). Actual learned confirmation behavior and exposure-controlled performance remain pending.

## Independent bounded-task evidence

`tools/audit_nfsp_bounded.py` verifies completed meaningful-branch and pure/mixed bounded-Go runs using the trial’s frozen package. It solves and certifies complete games independently, reloads every scheduled model, reconstructs eligible-Q/mode/reservoir/RNG clocks, and checks retained legal transitions and active frozen actors. See [the audit contract](docs/verification/nfsp-bounded-audit.md) for commands and explicit bounded-history limits. `tools/audit_nfsp_delayed.py` remains the separate chain verifier. A single case never completes A2.

## A2 cohort evidence

`tools/a2_cohort.py` creates a source-bound numerical plan, freshly audits an original completed case, and assesses all available cases against the fixed three-seed/final-five gates. Missing evidence remains incomplete. Matched controls and extension proposals are separate, and numerical passes do not authorize training. Archived reports can be reanalyzed without private model files. See [the protocol and commands](docs/verification/a2-cohort.md).

## Resource ledger

`tools/nfsp_resources.py` computes the design’s NFSP device-hour expression and reports separately recorded CPU/GPU scopes. Unknown prices stay missing, optional work requires an explicit selection, pilot cost is included in screening, and overlapping driver/phase measurements cannot be added twice. See [the resource contract and current recorded example](docs/verification/nfsp-resource-ledger.md). The calculator never qualifies a runtime forecast independently or authorizes training.

## Complete-component cost measurements

`tools/nfsp_cost_profile.py` reuses a sealed full-buffer corpus and measures real source kernels, actor-version stress, legal full-buffer append, complete IO and evaluation costs. It keeps materialized rows, benchmark updates and newly collected records distinct. Forecasts retain explicit event/game counts and margins, and profiling payloads are not public agents or runtime resumes. See [the measurement protocol](docs/verification/nfsp-cost-profile.md). Actual pricing/entry qualification remains separate.

The selected frozen-source R4 CUDA measurement and independent artifact/accounting audit are now [retained](docs/verification/nfsp-r4-full-cost/README.md). H_P is 52.744889 device-hours per seed and science/probe work is 76.616486 hours, exceeding their initial 24/16-hour components. The [updated partial ledger](docs/verification/nfsp-resource-ledger-measured-r4/README.md) records the measured H values and actual allocation; larger training remains unselected. Thirteen new cost-auditor tests pass separately from the latest complete 376-test suite.

## Explicit batched frozen evaluation

Frozen neural policies can select a fixed inference shape, with `--inference-batch-size 32`. Ordinary evaluation also uses `--pair-batch-size 16`; scientific evaluation uses `--pair-batching` within each original 48-game block. Cohort registration fixes the shape/device/pair count before training, and probe reloads recover the selected policy identity. Stable public-observation row placement preserves selected-policy marginals across single, grouped and resumed calls. See [numerical/persistence rules and verification](docs/verification/evaluation-batching/README.md). The full suite passes 403 tests; revised cost forecasts still require measurement.

The selected batched evaluator now has [complete measured costs and independent audit](docs/verification/nfsp-batched-costs/README.md), including actual two-look/48-game-block IO and cold setup. H_P is 56.236005 hours per seed and science/probes 24.819918, still above the initial 24/16-hour components. The corrected R4 table completes/audits all 15M records and misses supported credit/average readiness; its neural case continues. The [new partial resource ledger](docs/verification/nfsp-resource-ledger-batched-r4/README.md) settles the table conservatively and preserves all limits. Thirty-two affected tooling checks pass; the latest full package suite remains 403 tests.

The tactical baseline now reuses a [prepared complete single-placement resolver](docs/verification/prepared-placement/README.md). Frozen priorities and seeded choices are unchanged; all 420 tests and an exhaustive 3x3 differential check pass. The fixed CPU corpus measures about 11.5x faster tactical calls. Full selected-source pricing remains separate, and the original R4 neural case continues under its original limits.

The prepared-resolver source also completes its [full cost profile and independent audit](docs/verification/prepared-placement/full-cost/README.md): H_P **28.860612** hours per seed, with science/probes **24.940015**. Both initial component limits remain exceeded. [Declared resource use](docs/verification/nfsp-resource-ledger-prepared-r4/README.md) keeps the original control settlement and all prior work; no new training scope follows.

Frozen CUDA actor/evaluation forwards now support an explicit `--cuda-graphs` option. A bounded shared executor copies immutable weights/public inputs into fixed storage and replays the same forward; samplers and learning stay outside capture. All 430 tests pass, including exact outputs, private labels/RNG and checkpoint continuation. See [execution/provenance constraints and measured prototype](docs/verification/frozen-cuda-forward/README.md). The complete selected-source capture/packing costs and independent artifact audit are now [retained](docs/verification/nfsp-graph-costs/full-cost/README.md).

The original corrected R4 pair is now terminal at 15M records each. Its [fresh independent neural audit](docs/verification/nfsp-eligibility-R4-neural/README.md) confirms all five final credit/average reads pass; the matched table miss remains recorded. This is seed-zero delayed-family viability, with 34 paired-plan slots and global scope prerequisites still missing. No learner was restarted or reconfigured; later graph execution does not automatically inherit this source-bound A2 evidence.

The [completed cold/steady graph price audit](docs/verification/nfsp-graph-costs/full-cost/README.md) verifies H_P/H_D/H_N **18.852895 / 38.176705 / 131.821295** hours per seed, plus science/probes **14.923266** hours. The pilot/science forecasts now fit their unchanged 24/16-hour components. The [new resource ledger](docs/verification/nfsp-resource-ledger-graph-r4/README.md) preserves all prior declared use, failed preflight and distinct original neural costs. Missing C1/C4 prices, complete compatible A2 cohorts and a project ceiling still prevent a formal training scope.

`tools/task_source_diff.py` now records a source-pinned, read-only [old/current A2 dependency comparison](docs/verification/task-source-compatibility-ca45-c3b8/README.md). Twenty-four dependency modules, including package initialization, are byte-identical; the resolver, tactical and frozen-forward boundaries remain explicit. The eight new tests and sixteen existing resource checks pass. Behavioral compatibility and reuse of earlier A2 evidence remain unqualified.

The [trained actor-boundary check](docs/verification/task-source-compatibility-ca45-c3b8/actor-boundary/README.md) passes exactly on both original full 15M delayed-R4 archives under old/current packages, with no new training records. A separately bounded matched production-trainer prefix comparison is declared to check optimizer, buffer and trajectory state next. Complete source compatibility and A2 reuse are still pending.

The [matched complete task-pipeline checks](docs/verification/task-source-compatibility-ca45-c3b8/pipeline/README.md) now verify exact old/current model/optimizer, active-actor/RNG, replay/reservoir/metric and episode-byte equality across all eight diagnostic prefixes (524,288 total records). An explicit clock-field correction preserves the first checker failure and compares the same data without restarting training. Eighteen integrated tooling checks pass. The [settled resource ledger](docs/verification/nfsp-resource-ledger-task-parity/README.md) records 36,699.184125 committed control seconds; formal A2 cohorts and complete source qualification remain pending.
