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
| Shared PPO/GAE update backend | Implemented with terminal bootstrapping, eligibility, masking, clipping, entropy and KL stopping tests. Board rollout collection remains pending; matrix collection is verified. |
| Matrix real-trainer adapter | Implemented shared-update rollouts, current/history critic buckets, strict trial-plan resume, complete trainer-state recovery, and full preflight orchestration. Formal 12-trial gate completed and independently audited; neither protocol passed. Route: NFSP. See `verification/ppo-preflight.md`. |
| Run manifests and engine/network profiling | Implemented CLI and JSON reports. See `verification/phase-a.md` for recorded evidence and limits. |

## Selected-route infrastructure

- Packed replay/reservoir buffers, generation-safe sequences and resumable snapshots are implemented. The storage profile is deliberately partial; see `verification/buffers.md`.
- Complete-public-history independent episode-mixture evaluation, Bayesian prefix conditioning and optimal response are implemented and tested on a repeated-action counterexample and the reachable mixed Gray Go fixture.
- Fixed A2/9×9 exploration, logical-mode label eligibility and 5M–7M episode hand-off schedules are implemented. Marginal epsilon-greedy action probabilities include exploration hitting a greedy action.
- The formal PPO matrix preflight completed 24M aggregate records from a pinned source, with all 12 trials and reference controls. A separate weight-based audit confirms route **NFSP**. Unselected PPO bounded/delayed runs are skipped under the declared staged selection plan.

## Remaining work toward the full objective

1. Implement and verify the selected NFSP update/collector, true tabular NFSP controls and production-path resume. Reference exact-BR fictitious play is not a substitute for the Q-learning control.
2. Implement opaque delayed/branching tasks, balanced supplemental coverage, target/reuse comparisons and exact per-branch value/policy gates.
3. Run the selected NFSP A2 protocol across short mixed tasks, bounded Gray Go and the delayed family with three seeds, final-five gates and fixed aggregate counters.
4. Integrate the ninth-review development/confirmation cohorts, baseline confidence gates, opening checks and pilot safeguards.
5. Profile the complete selected learner, fill its cost ledger and run only the eligible declared pilot/screening/confirmation stages.
6. Implement protected evaluation/promotion, gray-rule/exposure comparisons, certified collision probes and required reports. Optional modeling/search remains conditional.

The Python engine is the independently readable reference. Its throughput must be measured before choosing a batched/native production backend. The reviewer prototype is used for differential evidence only. Generic checkpoint payload support and a restore integration test do not yet constitute a completed production training-loop resume protocol.
