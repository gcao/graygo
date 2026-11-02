# Corrected supplemental Q eligibility and record clocks

## Valid conditional targets

The collector now excludes only the passive player's pre-forcing root target when its opponent's choice is externally forced. The forcing player's own conditional Q(s,a) edge remains eligible, and both players' normally generated branch suffixes remain eligible. All supplemental labels stay outside the average-policy reservoir. No outcome/origin feature is added to the actor and no native value reference or threshold changes.

The stationary production-path regression initializes exact Q values for both controller orientations. Unrestricted replay retains the two excluded records for audit, but the Q batch loader rejects them. Eligible passive-root targets are exactly -1 and a production Q update leaves the exact value unchanged (zero loss). The forcing player's action-conditional targets remain +1/-1. This repairs the demonstrated kernel bias; it does not claim that every long-horizon approximation error or convergence problem is solved.

## Distinct freshness clocks

| Counter | Meaning | Drives |
|---|---|---|
| `fresh_records` | Every actual new own-side record, including excluded targets | Aggregate allocation, epsilon, opening, hand-off, evaluations and archive points |
| `fresh_q` | Actual value-eligible records | Q minibatch/reuse cadence and target refresh |
| `fresh_ordinary` | Ordinary, non-supplemental own-side records | Supervised publication cadence |

Q loss targets remain complete eligible minibatches, so reuse counts actual loss-bearing starts. Target copies occur after optimizer work at each 12,800 eligible-Q boundary. An excluded record at an already aligned boundary advances the aggregate budget but cannot repeat a Q update or target copy. Per-role aggregate/eligible counts and visit tensors are saved separately.

The randomized 8/1/1 horizon/controller allocation remains unchanged. For the complete balanced 15M plan, the 17,500 supplemental episodes would each exclude one passive-root target: 15M aggregate records versus 14,982,500 Q-eligible records, while ordinary/label clocks remain their own quantities. This is the arithmetic of the corrected plan, not a claim that a corrected 15M run has been collected.

## Joint commitments and exact boundaries

Both private actions and their frozen labels are computed before the resolver receives either action. Task collection then offers each role's real transition/eligible label and invokes the shared backend in chronological own-record order. An eligible boundary can be odd, so optimizer/target work may occur between storage of the two already committed roles. The other actor's current commitment cannot change in response. All stops, evaluation/archive writes and heavy checkpoints remain at complete paired-round boundaries; a partial joint round cannot be published as durable.

The board and ordinary matrix paths keep their batched collector/update path. The regression compares all-eligible per-record and paired updates, including online/target/average parameters, optimizer/counter state and replay RNG, and they match exactly.

## Replay and continuation

`sample_q` selects uniformly from retained value-eligible starts. One-step batch construction refuses missing/false eligibility. Q sequence context stops before an ineligible opponent-intervention transition; the prior valid nonterminal start can still bootstrap from its native next-state value. Unrestricted sampling remains available for complete record audits.

Replay metadata version two stores historical eligible insertions and current retained eligible size. Overwrites update live membership without erasing historical counts. Before wrap, a legacy file's complete retained mask history can be counted exactly. After wrap, missing historical eligibility remains explicitly unknown: read-only retained sampling works, but append/corrected continuation is refused rather than guessed. Original source-bound historical runs retain their original readers/continuation semantics.

Backend clock version two and task-state format version two make the changed semantics explicit. Checkpoints preserve aggregate/eligible clocks, per-role counts, eligibility visits, targets/optimizers, original frozen episode policies and all RNGs. CPU/CUDA tests restore at an even aggregate point with an odd eligible count and reproduce complete subsequent backend state, arrays, metrics, episode headers and private RNGs exactly. Task restore now preserves Adam CPU step tensors while placing model/optimizer data on their proper device.

## Independent evidence and remaining selection

The delayed-run auditor reconstructs excluded ordinals from episode headers and the actual forcing role, derives eligible counts at every scheduled archive/update, reconstructs the aggregate instant and live eligible sampler bound at each Q event, regenerates private RNG/Algorithm-R clocks and checks retained masks/per-role counters. It keeps independent closed-form native-opponent value grading. Completed bounded tabular/neural diagnostics pass; deliberately resealed wrong eligible clocks, native values and private RNG drift fail.

The original R4 comparison is terminal and independently audited under unchanged `9fefeee` source. Its neural average numerical band passes, while supported credit misses under the old defective eligibility rule (`nfsp-delayed-R4-neural.md`). Original source/grades remain preserved. No corrected convergence, 9x9 eligibility, target selection, extension or new large run follows from unit tests. Price the complete corrected implementation from matched pre/post profiles and record the corrective allocation before full training; old-source profiles cannot fund it.


## Verified test result

The final combined suite passes **288 tests in 183.616 seconds**. Eight focused eligibility regressions cover both stationary role orientations, unchanged valid own-action/suffix targets, loss rejection/filtering, nonrepeating invalid-boundary events, live/historical overwrite membership and sampler RNG, sequence truncation, all-eligible paired/per-record equivalence, CPU/CUDA complete-state continuation at an odd eligible clock, and refusal to guess or silently continue wrapped legacy eligibility history. The independent delayed audit also rejects validly resealed eligible-clock corruption, in addition to native-reference and private-RNG drift. Full original R4 data remain source-bound and unchanged. Corrected large-budget performance still requires its new matched profiles and recorded selection.
