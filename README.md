# Gray Go

Clean migration of the current Gray Go v6 self-play training implementation.

This repository contains the generic, de-versioned copy of the latest v6 code
from the old training tree. The main entry point is `run.py`; checkpoints are
written to `checkpoints/` by default.

## Layout

- `engine.py`, `engine.cpp`: Python and pybind11 game engines.
- `model.py`: neural network with a shared residual backbone, policy/value
  heads, and five auxiliary heads.
- `selfplay.py`: Python orchestration for joint-action MCTS self-play using
  the C++ callback MCTS engine.
- `selfplay_cpp.py`: pure C++/libtorch self-play wrapper for faster generation.
- `train.py`: replay buffer, online augmentation, and multi-head loss.
- `gate.py`: candidate-vs-champion promotion gate.
- `run.py`: end-to-end self-play, train, gate, save loop.
- `mcts_engine_callback.cpp`: C++ joint-action MCTS with Python model callback.
- `mcts_engine.cpp`: pure C++ self-play engine with libtorch inference.
- `export_model.py`: TorchScript export used by the pure C++ self-play path.

## Model Design

`GrayGoNet` is a player-relative residual network. The default production shape
is a 9x9 board, 12 residual blocks, and 128 filters.

### Input Encoding

Each position is encoded as `6 x S x S` planes from one player's perspective:

- Plane 0: current player's stones.
- Plane 1: opponent stones.
- Plane 2: gray stones.
- Plane 3: empty points.
- Plane 4: current player's forbidden points.
- Plane 5: opponent forbidden points.

The same network is used for both Black and White by re-encoding the board from
the requested player's perspective. Inference evaluates both player-relative
views and combines the scalar values as `(black_value - white_value) / 2`.

### Backbone

The shared body is:

- Circular 3x3 stem convolution from 6 channels to `filters`.
- BatchNorm and ReLU.
- `blocks` residual blocks.
- Each residual block has two circular 3x3 convolutions, BatchNorm after each
  convolution, and a ReLU residual add.

Circular padding matches the board's wraparound geometry and keeps spatial
features translation-stable across board edges.

### Heads

The network returns:

```text
(policy_logits, value, aux1, aux2, aux3, aux4, aux5)
```

- Policy: `1x1 conv -> BatchNorm -> Linear`, producing `S*S + 1` logits. The
  last action is pass.
- Value: `1x1 conv -> BatchNorm -> Linear -> Linear -> tanh`, producing a
  scalar from the encoded player's perspective.
- Aux1: spatial next-state stone delta, shape `S x S`.
- Aux2: scalar territory control ratio.
- Aux3: categorical opponent action distribution, shape `S*S + 1`, including
  pass.
- Aux4: scalar position complexity, sigmoid-bounded to `[0, 1]`.
- Aux5: spatial influence map, shape `S x S`.

The default 12-block, 128-filter model has about 3.63M parameters.

## Self-Play Design

Gray Go self-play is simultaneous-move: each turn chooses a joint action
`(black_action, white_action)`. MCTS searches joint actions, then visit counts
are marginalized into one policy target for Black and one policy target for
White.

### Per-Game Flow

For each game:

1. Start from an empty `GameState`.
2. For the first `randomize_first_n` turns, choose uniformly from legal actions
   for each player. The stored policy target is uniform over legal actions and
   aux4 is set to `0.5`.
3. After the randomized opening, run joint-action MCTS with model evaluation.
4. Convert joint visit counts into player-specific policy targets.
5. Sample the joint move from visit counts using `temp_high` before
   `temp_threshold` and `temp_low` afterward.
6. Step the game with both players' actions.
7. Store one training record for Black's perspective and one for White's
   perspective.

The run defaults are:

- `games_per_iteration = 300`
- `max_turns = 100`
- `mcts_visits = 600`
- `mcts_cpuct = 1.5`
- `mcts_tau = 0.01`
- `dirichlet_alpha = 0.15`
- `dirichlet_epsilon = 0.30`
- `temp_high = 1.0`
- `temp_low = 0.3`
- `temp_threshold = 15`
- `randomize_first_n = 4`

Long production runs can override these; for example the migrated training run
uses 350 games per iteration and 900 MCTS visits.

### Training Targets

Each stored sample contains:

- State: player-relative `6 x S x S` encoding before the joint move.
- Policy: marginalized MCTS visit distribution for that player.
- Value: final game outcome from that player's perspective.
- Aux1: stone delta between the pre-move and post-move state.
- Aux2: final territory margin from that player's perspective.
- Aux3: one-hot opponent action distribution, including pass.
- Aux4: normalized entropy of the player's marginalized MCTS policy.
- Aux5: BFS-based influence estimate from the pre-move state.

Aux5 treats own stones as `+1`, opponent stones as `-1`, gray stones as `0`,
and empty/gray regions by relative wraparound BFS distance to each color.

### Augmentation

Self-play applies spatial augmentation to generated samples:

- Two random D4 board symmetries.
- Two random torus shifts.

The pass action is preserved during spatial transforms. Spatial targets
(`policy` board part, aux1, aux3 board part, aux5) are transformed with the
board; pass logits/targets and scalar targets are preserved.

`train.py` also applies an additional random D4 symmetry and torus shift online
when sampling replay batches.

## C++ Self-Play Paths

There are two acceleration paths:

- Callback MCTS: `mcts_engine_callback.cpp` runs joint-action MCTS in C++ and
  calls back into Python for model evaluation. This is used by `selfplay.py`.
- Pure C++ self-play: `mcts_engine.cpp` loads a TorchScript model and runs
  self-play through libtorch. This is used when `run.py` is launched with
  `--use-cpp-selfplay`.

With `--use-cpp-selfplay`, `run.py` exports the current champion to
`checkpoints/champion_traced.pt` at the start of each iteration, verifies all
heads, runs C++ self-play, converts returned numpy arrays back into
`TrainingSample` objects, and then applies the same augmentation pipeline.

## Training Loop

Each `run.py` iteration does:

1. Generate self-play samples from the current champion.
2. Append samples to the replay buffer.
3. Copy champion weights into the candidate.
4. Train the candidate on replay data.
5. Gate the candidate against the champion.
6. Promote the candidate if gate win rate exceeds `gating_threshold`.
7. Optionally flush the replay buffer after a strong promotion.
8. Save champion, candidate, and JSONL metadata.

The default replay buffer holds 2M samples. The default gate is policy-only:
it samples directly from masked network policies rather than running MCTS. The
default gate threshold is `0.48`; the buffer flush threshold is `0.55`.

The training objective is:

```text
loss = policy_loss + value_loss
     + 0.50 * aux1_loss
     + 0.25 * aux2_loss
     + 0.30 * aux3_loss
     + 0.15 * aux4_loss
     + 0.20 * aux5_loss
```

Policy and aux3 losses are soft-target cross entropy over action
distributions. Value, aux1, aux2, aux4, and aux5 use MSE.

## Build

Build the Python game engine and callback MCTS:

```bash
make all
```

Build the pure C++ self-play path:

```bash
make cpp-selfplay
```

Run a CUDA training loop:

```bash
python run.py --resume --use-cpp-selfplay --device cuda
```

Generated checkpoints are written to `checkpoints/` by default and are ignored
by git.
