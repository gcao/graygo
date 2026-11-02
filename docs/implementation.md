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
- The full suite passes **125 tests**. See `verification/nfsp-backend.md`. This is implementation verification, not an NFSP convergence or 9×9 strength result.

## Resumable matrix NFSP trainer

`graygo train-nfsp` now runs neural or tabular native self-play, with shared parameters across roles and two fresh own-side records per physical game. It batches inference over the two public role observations and reuses the production Q loss, replay, masks, schedules and action sampler. Q, supervised and reservoir RNG streams are separate.

The tabular average policy is the exact conditional mean of the frozen greedy-label distributions currently in Algorithm R's reservoir, refreshed at the same supervised cadence. Integer counts by tie cardinality make tracker rebuilds independent of insertion order. Zero-support actions remain zero at deployment. A pure82 executed-action-label control uses the same behavior and changes only its averaging targets.

Resume bundles include all three models, both optimizers where applicable, buffers/cursors, clocks, RNGs, evaluations and the episode-log committed prefix. Tests reproduce future weights, replay/reservoir arrays, action logs and evaluations exactly after a partial-update pause. A per-episode binary audit retains mode, actual actor, both mode/ramp draws, ramp probability, epsilon, model generation, actions, behavior probabilities and terminal payoff. Checksummed prefixes detect corruption and remove only uncommitted crash tails on explicit resume. Writer locks and stale-checkpoint checks prevent overlapping progress.

Short diagnostic configurations cannot satisfy the formal-protocol flag. The default 15M-record primary recipe still requires three seeds, final-five checks, the other bounded/delayed/branching tasks and the complete cost ledger before A2 exit. No NFSP convergence or 9×9 strength claim is established by the trainer verification.

## Current learning evidence

Four full-path terminal-matrix timing diagnostics are recorded in `verification/nfsp-matrix-profile.md`. They include Q, averaging, audit, evaluation and checkpoint costs, with pre/post-opening scopes explicit. They do not price multiround or 9×9 work. The immutable first asymmetric screen plan is in `experiments/nfsp-asymmetric-primary-plan.json`; its source is isolated from subsequent development. Treat process/state files as observations, not gate results.

## Multiround NFSP integration

`train-nfsp-task` now integrates native simultaneous collection, frozen episode actors, replay/reservoir updates, complete resume and exact evaluation for opaque delayed/branching tasks and actual pure/mixed Gray Go roots. Bounded neural runs use the declared circular board encoder; tabular bounded controls index the complete public rule state. Delayed controllers/horizons use randomized 8/1/1 blocks with balanced round allocations. A 15M run over 50/100/200 rounds contains complete coverage macrocycles.

The checkpoints preserve active episode state, old frozen policy weights, mode/ramp choices, private action RNGs and trajectory history. Budget/chunk boundaries remain nonterminal. Coverage and replay counts are separate by public state, role, action and supplemental origin. Target/behavior overlap is measured using the same pre-update Q prediction as the loss, with mode/origin/fresh-age bins; no second diagnostic network forward changes the profiled work.

Exact delayed evaluation checks root action values and early target-policy values at depths 1/2/5 on both supported branches, with last-depth errors reported separately. Deployed average-policy gaps are graded separately from greedy-Q credit and readiness bands. Two-round Q value checks now condition the frozen native-opponent mixture on the observed public prefix; average/greedy game gaps remain against the original unregularized game. The branch control retains its meaningful mixed continuation matrices; no test threshold or game was simplified to manufacture a pass.

Full runs require an explicit time ceiling and matching declared complete-pipeline budget-profile hash. `purpose=profile` cannot satisfy formal gates. `tools/profile_nfsp_tasks.py` produces paired pre/post, R=1/R=4 costs; terminal-matrix costs are not substituted for these. Runtime settings, optimizer defaults, actual host memory/affinity/filesystem and complete adapter/codebook metadata are recorded.

The suite passes **125 tests**, including exact mid-episode resume for neural/tabular delayed and mixed-Go runs. See `verification/nfsp-multiround.md`. No delayed-credit convergence or 9×9 strength claim follows from these implementation checks.

The first complete delayed-pipeline profiler finished all eight cases. Its source is now historical for funding purposes after the native-opponent Q reference correction. The immutable eight-case plan and resource ceilings are in `experiments/nfsp-delayed-profile-plan.json`. Its results are not yet convergence evidence or a completed full-study budget.

## Remaining work toward the full objective

1. Independently audit completed matrix trials, finish the declared first-screen set (RPS and delayed family), then the required seed/task cohorts. Both seed-0 asymmetric primary runs completed 15M each and passed the independent archive/RNG/reservoir/replay audit. See `verification/nfsp-asymmetric.md`. This does not complete the other families/seeds.
2. The corrected-reference delayed profiles are complete and C1/C4 costs are entered; run the selected R=1 first screen, profile bounded neural/control paths, and run the selected one-step comparisons with actual tabular controls. Per-branch gates and full multiround resume are implemented; learning evidence and any conditional Retrace selection remain pending.
3. Run the selected NFSP A2 protocol across short mixed tasks, bounded Gray Go and the delayed family with three seeds, final-five gates and fixed aggregate counters.
4. Integrate the ninth-review development/confirmation cohorts, baseline confidence gates, opening checks and pilot safeguards. Keep packed-feature storage's current overhead explicit; the design's compact rule-state/regenerated-feature backend is still pending.
5. Profile the complete selected learner, fill its cost ledger and run only the eligible declared pilot/screening/confirmation stages.
6. Implement protected evaluation/promotion, gray-rule/exposure comparisons, certified collision probes and required reports. Optional modeling/search remains conditional.

The Python engine is the independently readable reference. Its throughput must be measured before choosing a batched/native production backend. The reviewer prototype is used for differential evidence only. Both matrix and finite multiround trainers now have verified complete resume protocols. Large-board collection optimization, compact rule-state storage, pilot/confirmation orchestration and their production-scale verification remain pending.

## Reference-value correction before delayed learning

Q credit is now evaluated for its own target policy against the actual native opponent episode mixture, including posterior conditioning after public prefixes. This keeps averaging lag separate from credit learning. The previous joint-epsilon-Q assumption was an implementation error, corrected before full delayed comparisons. See `verification/nfsp-native-reference.md`. Historical timing measurements are retained and their actual CPU cost remains in the record; they cannot fund the changed source.

## Current allocations after audited asymmetric results

The source-corrected delayed cost report and partial actual-resource ledger are in `verification/nfsp-delayed-costs-9fefeee.md` and `experiments/resource-ledger.md`. RPS cost profiles passed their explicit gate and its seed-0 primary screen is live. The delayed driver permits only the budget-bound R=1 control/neural first screen; R=4 is not automatically selected by profiling it. All one-seed results remain partial A2 evidence.
