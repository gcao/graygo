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

## Selected NFSP backend and controls

- `nfsp.py` implements expected one-step targets using online epsilon-greedy probabilities and frozen target-network values. Terminal successors do not bootstrap; continuing replay cuts do. Targets are projected to `[-1,1]`, and raw excursions are reported.
- Q and average policy have independent encoders/optimizers. The board average policy has no unused PPO critic. Masked soft-target cross-entropy preserves frozen greedy ties.
- Fresh Q/ordinary counters drive reuse, warm-up, supervised cadence and 12,800-record hard refresh. Actual replay insertions must match the Q clock. Target refresh follows the boundary's optimizer work and remains fixed when reuse changes.
- `nfsp_collection.py` freezes actor parameters and epsilon per episode. Logical mode, acting policy, version, marginal behavior probability and supplemental origin are separate. Both commitments precede transition resolution. Forced coverage includes the actual root edge; its entire episode is excluded from reservoir labels.
- Opaque 50/100/200-round controls, a forced-action indexing variant, a 40-equivalent-action stress variant and a meaningful two-round branching game are implemented. Codebooks are randomly permuted independently of semantic order, with exact manifest/digest support.
- Exact finite Markov solving, policy/action-value evaluation and best responses are implemented for those adapters. Latent episode mixtures continue to use the complete-history evaluator.
- Supported-coverage scheduling gives each horizon/controller equal rounds, with randomized ten-episode blocks containing eight ordinary and one episode per forced branch. Mid-block and mid-episode resume checks pass.
- The full suite passes **114 tests**. See `verification/nfsp-backend.md`. This is implementation verification, not an NFSP convergence or 9×9 strength result.

## Resumable matrix NFSP trainer

`graygo train-nfsp` now runs neural or tabular native self-play, with shared parameters across roles and two fresh own-side records per physical game. It batches inference over the two public role observations and reuses the production Q loss, replay, masks, schedules and action sampler. Q, supervised and reservoir RNG streams are separate.

The tabular average policy is the exact conditional mean of the frozen greedy-label distributions currently in Algorithm R's reservoir, refreshed at the same supervised cadence. Integer counts by tie cardinality make tracker rebuilds independent of insertion order. Zero-support actions remain zero at deployment. A pure82 executed-action-label control uses the same behavior and changes only its averaging targets.

Resume bundles include all three models, both optimizers where applicable, buffers/cursors, clocks, RNGs, evaluations and the episode-log committed prefix. Tests reproduce future weights, replay/reservoir arrays, action logs and evaluations exactly after a partial-update pause. A per-episode binary audit retains mode, actual actor, both mode/ramp draws, ramp probability, epsilon, model generation, actions, behavior probabilities and terminal payoff. Checksummed prefixes detect corruption and remove only uncommitted crash tails on explicit resume. Writer locks and stale-checkpoint checks prevent overlapping progress.

Short diagnostic configurations cannot satisfy the formal-protocol flag. The default 15M-record primary recipe still requires three seeds, final-five checks, the other bounded/delayed/branching tasks and the complete cost ledger before A2 exit. No NFSP convergence or 9×9 strength claim is established by the trainer verification.

## Remaining work toward the full objective

1. Extend the implemented resumable matrix NFSP trainer to multiround task/board collection, full episode/buffer checkpoint bundles and per-branch evaluation. The neural and actual tabular reservoir-average matrix paths, shared loss/sampler/counters and CLI are implemented. Their formal learning runs remain pending profiling.
2. Wire the implemented opaque/branching tasks and balanced supplemental schedule into production-path training/evaluation. Run the one-step reuse comparison and exact per-branch Q/greedy and average-policy gates before selecting any conditional Retrace work.
3. Run the selected NFSP A2 protocol across short mixed tasks, bounded Gray Go and the delayed family with three seeds, final-five gates and fixed aggregate counters.
4. Integrate the ninth-review development/confirmation cohorts, baseline confidence gates, opening checks and pilot safeguards. Keep packed-feature storage's current overhead explicit; the design's compact rule-state/regenerated-feature backend is still pending.
5. Profile the complete selected learner, fill its cost ledger and run only the eligible declared pilot/screening/confirmation stages.
6. Implement protected evaluation/promotion, gray-rule/exposure comparisons, certified collision probes and required reports. Optional modeling/search remains conditional.

The Python engine is the independently readable reference. Its throughput must be measured before choosing a batched/native production backend. The reviewer prototype is used for differential evidence only. Generic checkpoint payload support and a restore integration test do not yet constitute a completed production training-loop resume protocol.
