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

The repository contains the immutable reference engine, full-state encoding, random/tactical baselines, policy/value/Q networks, finite-game solvers, checkpoint storage and recovery, and CLI commands. The completed PPO matrix preflight selected NFSP; its independently audited evidence is in [the preflight report](docs/verification/ppo-preflight.md). The shared NFSP update/collection backend, compact buffers and solved delayed/branching controls are implemented. The NFSP campaign driver, convergence gates and scientific comparisons remain in progress; synthetic updates are not trained-game results.

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

The one-step backend in `nfsp.py` is shared by compact task and board models. `nfsp_collection.py` provides frozen per-episode native self-play and independently sampled simultaneous actions. `tasks.py` supplies opaque delayed/branching controls and balanced value-only coverage; `public_game.py` computes exact fixed-policy/action values and responses. Training-run orchestration and the tabular reservoir-average control remain required before formal NFSP A2 runs.

```sh
.venv/bin/python -m unittest discover -s tests -v
# Partial storage benchmark; includes checksummed save/load, not full learner cost.
.venv/bin/graygo profile-buffers --buffer-dir artifacts/buffers --records 10000
```

See [NFSP backend verification](docs/verification/nfsp-backend.md) for the checks and their limits.
