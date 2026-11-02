# Foundation verification

This records implemented foundation behavior. The full implementation goal remains active; production rollout adapters, replay/reservoirs, learning gates, and the scientific training campaign remain outstanding in `../implementation.md`.

## Reproducible evidence

- Engine/model foundation commit: `77f34e6665e8e4d890367e95d8c514a8a0e8432c`.
- Shared PPO/GAE backend commit: `42ff4942c4a20b52cecfe8d9587e4f08fc2ca61d`.
- Integrated suite: **47 tests passed**, with no skips. See [test output](phase-a/tests.log).
- All three benchmark manifests record a clean foundation commit, source hash, design hash, rule/encoding/baseline versions, seed, dependencies, host, and device information.

| Check | Observed result |
|---|---|
| Differential audit | 3,000 joint rounds, 15 completed games, zero transition/reward/encoding mismatches against the external prototype. |
| Reference-engine workload | 100,000 joint rounds in 24.847 seconds, 4024.6 rounds/second; 559 completed games. |
| Full network | 630,787 parameters, 64 channels, eight residual blocks, CUDA batch 16. |
| Synthetic network inference | 10 batches in 0.025188 seconds. |
| Synthetic network updates | 10 batches in 0.354257 seconds; peak allocated CUDA memory 48,161,792 bytes. |

Raw reports: [engine](phase-a/engine-profile.json), [network](phase-a/network-profile.json), [native comparison](phase-a/native-audit.json).

The final benchmark sequence ran serially. The native audit uses one process-scoped allocation because the reviewer ABI has no destructor. Its external library hash is recorded in the report. That comparison is supplementary evidence; independent hand-authored fixtures and complete-state symmetry tests also check the rules.

## Rules and state coverage

Tests cover toroidal groups; newborn-group protection during capture; simultaneous suicide; gray capture in both directions; disappearing collisions; forbidden-point accumulation/clearing; outgoing-move ko and expiration; pass streaks and the round limit; exact quarter-point allocations; full-state serialization; and geometric/color commutation. Capture fixtures include the owner examples.

Encoding tests cover current occupancy/liberties, both saved boards and move pairs, restrictions, missing-history/pass flags, countdown, perspective exchange, legality, terminal masks, and immutability.

## Solved games

The LP solver checks primal/dual security bounds and refuses large residual gaps. Exhaustive solving has a hard state budget and never substitutes heuristic leaves on exhaustion. Best responses currently support fixed public-state Markov policies; complete-history episode-mixture evaluation is still required for its later gate.

The reachable fixture in `tests/fixtures/mixed_subgame.json` has two rounds left on a 3x3 research board. Its exhaustive continuation matrix is:

```text
      white 0   8    pass
black 0   -1    0      0
      8    0   -1      0
   pass    0    0     -1/2
```

Both equilibrium policies are `(1/4, 1/4, 1/2)`, with value `-1/4` to black. Prefix replay, matrix values, and independent best-response recursion are checked. This supplies a real mixed-strategy game rather than assuming any small position needs mixing.

## Persistence and learning backend

Storage tests cover projected disk reserve, checksums, active-writer exclusion, process death, recovery after generation commit but before pointer publication, rotation/finalization, and permanent weights. An integration test restores actual model/optimizer, Python/NumPy/Torch random streams, and environment state. Production replay/reservoir and collector-resume integration remains outstanding.

PPO tests cover terminal/chunk bootstrapping, episode boundaries, actor/value eligibility, masked actions, policy-improvement direction, fully excluded records, finite gradients, and KL stopping. This is shared update machinery; no PPO matrix preflight has run.

## Limits

The Python engine is a correctness reference, and its timing does not establish native/batched training throughput. CUDA updates are synthetic policy/value work, not trained agents, NFSP replay, or a full-campaign cost. The learning gates and ninth-review protocol decisions remain ahead of the pilot. No learned playing-strength claim is made.
