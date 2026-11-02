# NFSP backend and solved-control verification

The combined suite passed **100 tests in 11.756 seconds** on `ai`. `nfsp-backend/tests.log` records the complete output; `nfsp-backend/manifest.json` records hardware, dependencies and source identity. This evidence covers the backend and adapters, not a completed NFSP convergence experiment.

## Checks with independent expectations

- One-step bootstrap uses `pi` chosen from online Q and evaluates it with frozen target Q. A hand-calculated fixture gives `0.95*0.2 + 0.05*0.8 = 0.23`; terminal reward is unchanged, illegal high actions are ignored, and a continuing buffer boundary retains its bootstrap.
- Q learns the executed exploratory action. Frozen greedy-label distributions remain separate, and supervised masking gives finite gradients without assigning illegal mass.
- `R=1` and `R=4` have identical 12,800-fresh-record target clocks. Warm-up and ordinary-record supervised clocks are distinct. Empty/closed reservoirs create no backlog.
- Backend weights, both optimizer states, counters, replay/reservoir samples and sampler RNG reproduce the same next updates after restore. Episode restore preserves frozen actors, private action RNGs and the resulting trajectory. The existing checksummed storage integration independently covers both compact buffers.
- A full default 64-channel/eight-block 9×9 Q update and a separate default average-policy update complete with finite losses/gradients. This one-sample CPU smoke is not a throughput measurement or learned strength result.
- The forced-action chain has known optimal own utility `+1`, supported root-action values `(1,-1)`, and both 50/100/200-round role orientations. The 40-action variant provides genuinely equivalent joint continuations.
- The two branching continuation matrices have exact controller values `-1/7` and `-7/22`; the root prefers the first. Role exchange negates/transposes payoffs. Uniform-continuation root action values are `(-0.125,-0.3125)`. A later action can change the result from `-1` to `+1`, exposing why an uncorrected behavior suffix need not evaluate a target continuation.
- Opaque codebooks expose one shuffled one-hot ID plus countdown, with no outcome/branch scalar. Coverage macrocycles allocate 2,000 rounds to each horizon/controller, within randomized 8/1/1 episode blocks; partial-block resume is exact.
- Invalid terminal masks, nonfinite/out-of-range values, shaped nonterminal rewards, forbidden labels, clock/storage mismatch, shared player RNGs and exhausted solver budgets fail explicitly.

## Scope limits

No NFSP A2 learning run, target/reuse selection, tabular averaging-control result, complete-learner cost profile, 9×9 pilot or strength claim is established here. The tabular Q model supplies direct projected estimates; the required exact reservoir-average tabular policy and its actual control runs remain to be integrated. Retrace is conditional on the one-step comparison and has not been selected. Packed observations are a correctness-preserving development representation; the design's compact rule-state storage and regenerated features still need implementation/profiling.
