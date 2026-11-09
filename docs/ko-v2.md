# Gray Go v2: full-game ko

Owner-approved rule change, November 9, 2026. This document supersedes the
two-round ko rule and the claim that the fixed neural observation is the complete
rule state **for v2 only**. Existing v1 games, evidence, checkpoints and campaigns
retain their original rules.

## Resolution

1. Both players commit legal moves against the same pre-round board and their
   own forbidden masks. PASS is always legal.
2. Tentatively place stones, resolve collisions, capture existing dead groups,
   then remove remaining suicidal groups. Capture order is unchanged.
3. Unless both moves were PASS, compare the tentative stone board with **every
   committed stone board in this game**, including the initial board.
4. On a match, reject the entire joint result. Keep the pre-round board and commit
   no placements, collisions or captures. Add both attempted non-pass points to
   their respective forbidden sets at this current position. Existing bans remain.
   This attempt consumes one round and resets the mutual-pass streak to zero.
5. On a new board, commit it, remember it and clear the previous forbidden sets.
   Restrictions accumulate across any number of unchanged rounds; PASS/PASS
   preserves them.

An unchanged suicidal attempt is a repetition of the current board and follows
step 4. Repetition compares stones only, ignoring counters, masks and move history.
Only actual PASS/PASS rounds count toward termination: two consecutive such rounds
finish the game. The existing scored 200-round limit still applies, including
rejected attempts. There is no special forced draw for repetition.

The recorded four-round loop from the reported game now blocks its fourth joint
attempt. The board remains at the third position; tentative recaptures are undone.
Black's i1 and White's f3 become forbidden. Under whole-game replay, an earlier
repeat in the same export is first blocked at zero-based turn 25.

## Network and training

The 30 spatial channels, 14 globals and network weights are unchanged. The engine
enforces full-game repetition; the model consumes the existing current board,
forbidden masks and two recent rounds. This is now a **partial observation**, not
a complete Markov rule state. Current masks describe individually banned actions,
but cannot predict every future joint repetition. Existing weights can play v2;
strength and calibration under the new rule require new evidence.

Use `Rules(ko="full-history")` explicitly. Ordinary `Rules()` and existing training
plans remain v1. v2 states serialize their complete distinct-board history;
active collector checkpoints restore it. Rotation, reflection, torus shifts and
color swaps transform that history too.

Compact replay packets deliberately store only the unchanged observation fields.
Their fingerprint and replay descriptor bind `graygo/2`, separating them from v1.
Decoding a v2 packet produces an observation-only state that cannot be simulated
or saved as a complete game. The native decoder ABI and tensor dimensions remain
unchanged.

For a new v2 campaign, `BoardRuntimeConfig(initial_weights_archive=...,
rules=Rules(ko="full-history"))` verifies the sealed source archive and copies
only Q and average network weights. The target starts from Q. Optimizers, update
counts, replay and supervised reservoir all start fresh. The imported average is
immediately usable without inventing supervised-update counts. Provenance records
the source manifest hash and both rule configurations. A v1 checkpoint cannot be
resumed as a v2 runtime; do not mix v1 replay or historical v1 match results into
v2 gates. Evaluate historical weights under v2 rules using the explicit
`evaluation_rules` transfer, also supported by `evaluate --ko full-history`.

The production v1 loop remains pinned to its original worktree. This rollout
does not reset or change it. Switching production requires a separate v2 campaign
and freshly collected evaluation evidence.

## Verification

- Shared four-round fixture and the sanitized 91-round user export: v1 replay is
  unchanged; v2 blocks the first repeated result.
- Per-player cumulative bans, collision/suicide rejection, actual mutual-pass
  termination, full-history serialization and checkpoint continuation.
- Python/native observation parity and weights-only initialization.
- Differential trainer/arena checks: 5,446 rounds over 3x3, 5x5 and 9x9 boards,
  including 190 rejected v2 attempts, with matching boards, masks, counters,
  histories and scores.
- Production-sized candidate-59506688: 400 fresh v2 records, zero optimizer
  updates, identical Q/average tensor hashes, complete collector history restored.
- v1 state serialization, neural features, legal masks and compact packets match
  the pinned production source across all 91 recorded rounds.

The bounded validation is an integration check, not a strength benchmark or
promotion. Validation artifacts are under
`ai:~/proj/graygo/runs/ko-v2-validation-20261109/`.
