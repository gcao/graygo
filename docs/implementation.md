# Implementation status

Objective: implement Gray Go following `docs/design.md`, including its verification and learning gates. This file records progress; it does not replace or narrow that objective.

## Foundation

| Design requirement | Current evidence/status |
|---|---|
| Immutable full public rule state and deterministic joint resolver | Implemented in `engine.py`; hand-authored capture, suicide, ko, history, pass, round-limit and scoring tests. |
| Toroidal topology and exact quarter-point totals | Implemented; wrap/group and all territory-boundary allocation tests. |
| Complete-state geometric/color symmetry | Resolution commutation and scoring tests across shifts, rotations, reflections, and color exchange. |
| Complete 30+14 feature encoding and action masks | Implemented with fixed oldest/newest history slots; perspective and missing-history tests. |
| Frozen reproducible baseline agents | Random and tactical v1; tactical effects use hypothetical opponent pass, randomized ties, capture/save/safe/pass priorities. |
| Policy/value/Q architecture | Circular 64-channel/eight-block encoder, masked policy, bounded critic/Q, critic-only bucket input. Shape/gradient/equivariance tests and a full-network CUDA synthetic profile. |
| Checkpoint retention and disk reserve | Checksummed atomic bundles; two active resume generations, final-generation retention, permanent weight archive, 20 GiB projected guard. Interrupted-process, corruption/fallback, and real model/optimizer/RNG/environment restore tests. |
| Exact bounded-game tools | Matrix LP certificates and exhaustive public-state solver with explicit node budgets. Matching pennies, asymmetric matrix, RPS and actual one/two-round Gray Go fixtures. |
| Genuinely mixed Gray Go fixture | Reachable 3×3 position, two rounds remaining; root continuation matrix `diag(-1,-1,-1/2)`, value `-1/4`, both policies `(1/4,1/4,1/2)`. Prefix replay and independent best-response checks. |
| Shared PPO/GAE update backend | Implemented with terminal bootstrapping, eligibility, masking, clipping, entropy and KL stopping tests. Rollout collection/adapters remain pending. |
| Run manifests and engine/network profiling | Implemented CLI and JSON reports. See `verification/phase-a.md` for recorded evidence and limits. |

## Remaining work toward the full objective

1. Implement compact replay/reservoir buffers and rollout collection around the shared PPO updater; matrix adapters must use that machinery.
2. Implement exact complete-history/episode-mixture evaluation, tabular controls, delayed/branching tasks and their supported-coverage masks.
3. Run the prescribed PPO real-trainer preflight with three seeds and both entropy protocols. Record actual outcomes and select the route from evidence.
4. If required, implement/verify the NFSP-style learner, target/reuse comparisons, greedy labels, opening protocol, and deployed-policy gates.
5. Resolve the ninth review's hand-off, seed-selection, screening-baseline and fixed-epsilon details explicitly before the NFSP pilot. The current design body predates that review; implementation must not silently substitute a protocol.
6. Profile the complete selected learner, fill its budget ledger, and run the declared pilot/screening/confirmation stages with reproducible checkpoints.
7. Implement protected evaluation/promotion, gray-rule/exposure comparisons, certified collision probes and the required reports. Optional modeling/search work remains conditional on results.

The Python engine is the independently readable reference. Its throughput must be measured before choosing a batched/native production backend. The reviewer prototype is used for differential evidence only. Generic checkpoint payload support and a restore integration test do not yet constitute a completed production training-loop resume protocol.
