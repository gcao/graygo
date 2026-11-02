# Independent meaningful-branch and bounded-Go audit

`tools/audit_nfsp_bounded.py` audits completed `branching`, `gray-pure` and `gray-mixed` runs from the original task manifests, sealed learner/buffer checkpoints, binary episode headers and every permanent scheduled model archive. The existing chain auditor and all source-pinned historical copies remain unchanged. Each audit uses the trial's frozen package and exact dependency versions. A single task receipt never grants A2 completion or a production training scope.

## Independent game and policy calculations

The auditor enumerates complete legal finite-game trees from the actual public roots and engine transitions. Floating matrix LP output proposes strategies; exact Fraction arithmetic checks normalized nonnegative strategies and equal primal/dual security bounds at every nonterminal public state. No restricted action subset or approximate leaf value enters this proof. The receipt includes every exact continuation matrix, complete legal action indices, rational equilibrium strategies/security values, original public states and a checksum binding the whole game certificate. These are stage matrices with certified continuation values, not a claim that a root matrix alone represents all future choices.

The meaningful-branch roots have exact values **-1/7 and +1/7**. The pure bounded-Go root has value **0**; the actual reachable two-round mixed-Go root has value **-1/4**. These are independent game-value checks, not trained-policy scores. The mixed game's certificate contains nontrivial mixed strategies. Root/state inventories must match the original adapter, including opaque codebooks and complete board history/restrictions.

Average and greedy-Q policy gaps come from independent complete dynamic best responses to the deployed Markov policies. The native Q-value check independently evaluates fixed opponent components first, then combines their values using Bayes weights from revealed public opponent moves. It does not call the trainer's task evaluator, its native-value recursion or its policy-gap function. This matters at the meaningful root: a revealed action changes the posterior probability of the episode-frozen Q/average/uniform opponent, so its initial mixture weights cannot simply be reused downstream.

All scheduled model archives are reloaded. The auditor recomputes both roles' gaps and current-target supported values at every required state, then checks the stored per-root rows and final-five decisions. The final public Q/average tensors must equal the final learner tensors. Branching requires the supported-value and greedy-Q criteria as well as the short-game average criterion; bounded-Go short-game acceptance uses its average policy. Reduced diagnostic settings remain nonformal.

## Collection, storage and recovery checks

Header reconstruction independently replays balanced 8/1/1 coverage where applicable, per-player mode/ramp draws, Q fallback versus actual acting policy, aggregate/ordinary/eligible-Q clocks, passive forced-root exclusions, Q reuse and target refresh, supervised cadence, Algorithm-R acceptances/replacements and private action/sampling RNG continuation. It checks optimizer steps, per-role and state/action/origin metric totals. Normal board episodes do not acquire supplemental coverage or forced roots.

Every retained replay observation is decoded to a complete legal public state at its original root/depth/role. Both commitments must refer to the same public state; actual paired actions are passed through the game engine to check the complete next observation, terminal reward and ending. An odd diagnostic capacity can retain one own-side record without its paired opponent action; the receipt counts that reachable-transition check separately rather than calling it a fully observed joint transition. Formal 500k capacity retains complete pairs.

Retained reservoir records must originate after opening from ordinary logical best-response visits, retain the expected frozen version and contain nonempty legal greedy sets. For tabular publication, unchanged historical labels reproduce exact conditional means. Replacements after the last supervised publication can discard an old label/observation. The auditor reverses their record-slot history, lists affected tokens and retains necessary simplex interval bounds for unresolved labels. It verifies unaffected rows exactly and reports whole-table verification false when any old published label remains unresolved. Interval feasibility is not a proof of a joint historical assignment.

Active episodes receive stronger checks because their frozen models and original logs survive: the auditor reloads those models, reconstructs private inverse-CDF action draws and float64 probabilities, verifies retained frozen greedy targets where available, and checks actor/global RNG and full public-state continuation. CPU verification preserves caller CPU/CUDA RNG states and thread settings, including an ambient CUDA default device. It holds a shared nonblocking lock on the existing trainer lock for the entire read, and creates no training directories or snapshots.

## Explicit history limits

Bounded replay stores behavior probabilities as float32. Completed episode hashes cover original float64 probabilities. Retained legal action paths can be reconstructed, but the discarded probability bits and corresponding original log hashes cannot be recovered. Earlier actions and targets overwritten by bounded buffers also remain unavailable. The receipt states these limits; it does not claim to reconstruct every historical gradient update, action log or greedy label. Code tests, exact archived grades and retained-data checks remain distinct from such a full-history proof.

## Command

```sh
PYTHONPATH=FROZEN_TRIAL/src .venv/bin/python tools/audit_nfsp_bounded.py RUN \
  --budget-profile ORIGINAL_MATCHED_BUDGET.json \
  --time-cap-seconds 600 --output artifacts/bounded-audit.json
```

The default demands the formal primary protocol. Use `--allow-diagnostic` only for explicitly reduced diagnostic runs; it does not relabel their settings as formal. The watchdog defaults to one hour and cannot exceed four elapsed hours. It limits one audit invocation and supplies no new aggregate CPU-stage allocation. A timeout writes no success receipt. Measured audit-entry wall time is recorded; interpreter/import, tests, development and other overhead remain additional. Output must lie outside original training inputs.

## Verification status

Targeted tests cover both tabular and neural learners on all three bounded families, rule-state packets, exact roots, component posterior updates, real CLI output protection, complete/active paths, resealed value/action/RNG/clock corruption, missing archives and codebooks, dependency drift, tabular publication uncertainty, odd capacity, exclusive-writer exclusion, caller RNG/thread preservation, R4 reuse and the uniform comparator. The old chain auditor remains the independent verifier for the live corrected R1 trial. Formal three-seed bounded/branching training evidence and the complete A2 exit decision remain required.

The final combined suite passes **320 tests in 248.580 seconds**, including nineteen new bounded-audit test methods and both-learner/all-family subcases. The [full regression log](nfsp-bounded-audit-tests.txt) and [source/validation record](nfsp-bounded-audit-validation.json) are retained. These are reduced diagnostic fixtures and proof checks; no formal A2 cohort or additional target/learner experiment is selected.
