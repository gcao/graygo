# Prepared single-placement resolver and frozen tactical baseline

The measured B=32 pilot forecast is dominated by the tactical baseline and independent frozen actor calls. Tactical v1 previously ran the complete generic resolver independently at every legal candidate, then scanned every resulting board again. The new `PlacementResolver` reuses one analysis of the same immutable pre-round state and still returns a **complete actual game transition**. Tactical priorities and private tie sampling remain unchanged; no new baseline policy or game rule is introduced.

## Exact engine specialization

On a resolved board there are no zero-liberty groups. With one placement and an opponent pass, only groups adjacent to the placement lose a liberty. All adjacent friendly groups join the new stone. Adjacent nonfriendly groups whose sole old liberty is that point are exactly the captured groups; their row-major order is retained. This includes gray captures. After those captures are removed, the merged friendly group's actual liberties determine survival/suicide. Removing groups only adds liberties elsewhere, so no other group becomes dead.

The prepared resolver returns the complete board/capture/suicide outcome, next-player restrictions, two-round ko/history, pass streak, countdown and actual terminal reward. It also exposes the surviving placed group so the tactical agent does not need another whole-board scan. Full reward scoring still runs at terminal states. Both-pass, collisions and two distinct placements continue through the retained general resolver. The public `resolve` uses this exact specialization only for canonical one-placement/pass calls; prepared callers can reuse its analysis across many candidates.

Internal state assembly avoids rechecking invariants already proved by a canonical parent and the capture/removal algorithm. It creates immutable canonical tuples/frozensets and normalized integral actions. External `State`/`RoundRecord`/packet construction continues validating input. Differential checks compare every transition field and revalidate constructed states through the public schema. No unchecked external data path is added.

The original resolver body is retained as `resolve_reference`, with no statement changes. The original tactical body is retained as `_tactical_reference`, bound to that independent resolver. Portable canonical AST fingerprints match the frozen pre-optimization bodies. This preserves an oracle that does not call the optimized engine path. Baseline version stays `graygo/baselines/1`; source hashes change as required. Old scored results and prices are never relabeled as this implementation.

## Verification scope

Eight targeted tests pass. They compare complete transitions, placed groups, every tactical rank, selected actions and post-choice private RNG state on all 64 frozen dense roots, actual histories/restrictions, suicide/capture/ko fixtures, terminal rounds, both collision rules, wrapping/color transformations and 13x13/19x19 boards. Reference bodies are checked against frozen fingerprints. Paired games and episode mixtures preserve their complete scored results and trajectories.

A core-only CLI subprocess deliberately blocks Torch imports and completes batched tactical/random evaluation. Batched baseline-only evaluation now keeps the core NumPy dependency contract; neural evaluation still initializes its selected deterministic runtime.

The complete package suite passes **420 tests in 293.408 seconds**, with unchanged source `61c271fb94bd421b47aeb941fb43e03dcac307fd11a488720c0d560710d10eed`. Original suite driver/test PIDs **2613950 / 2613974** are terminal; [the complete log](suite/tests.txt) and [job](suite/job.json) are retained. A separately [allocated exhaustive CPU check](../../experiments/prepared-placement-verification-allocation.json) enumerates all **524,288** 3x3 boards across both collision-rule contexts, rejects unresolved dead-group roots, and checks both roles/all legal one-sided/pass transitions and exact tactical/RNG agreement. These are hypothetical verification states, not learning or strength evidence. Larger complete states and nonzero history/restriction metadata remain separately covered by tests.

The exhaustive original driver/checker **2614923 / 2614924** are terminal and verified in 287.871262 driver seconds. The [report](exhaustive/report.json) enumerates all **524,288** board/rule contexts: **340,094 resolved states**, 184,194 rejected dead-group roots, **2,609,860** complete transition/placed-group checks and **680,188** exact tactical-choice/post-RNG comparisons. Source/tool bytes and the literal oracle fingerprints are unchanged.

The matched CPU corpus contains 124 tactical calls per repetition, generated from every eighth nonterminal state of three fixed-seed reference/uniform official games. Three repetitions alternate implementation order. Median prepared/reference intervals are **0.033921 / 0.390953 seconds**, about **11.525x** on this finite corpus, with identical private-choice digest. This is implementation timing, not a universal speedup, full learner price or trained-game result. The [original job](exhaustive/job.json), [driver](exhaustive/driver.py), source plan/manifest and report preserve the measured scope.

## Remaining cost qualification

Existing H_P/H_N/scientific prices belong to their original frozen source. The CPU timing result cannot replace a full learner/evaluation price. Selected-source complete hardware measurement and independent resource accounting are still required before a feasible production scope. The original corrected R4 neural learner remains on its original process, frozen source, full record cap and time ceiling. No A2 gate or larger training allocation is granted here.
