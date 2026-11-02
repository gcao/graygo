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
