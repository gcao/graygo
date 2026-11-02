# Version-two controlled probe construction

## Purpose and preserved requirements

The independently reproduced one-cell constructor stopped at its declared cap with 24 distinct complete-game classes. Version two adds a second controlled occupied-cell removal/recoloring to explore group/liberty/score interactions absent from that catalog. It starts a separate preregistered search; it does not extend the stopped run, combine catalogs opportunistically or count the same game under another label.

The game rules, default 9x9/200-round root semantics, exact individual gap 1/20, range separation 1/10, collision shift 1/20, full legal action set, complete-game isomorphism test, root reuse rejection, alternating direction schedule and required 32-family coverage are unchanged. No trained-policy output enters proposal, filtering, orientation or selection. Exact equilibrium teacher fixtures remain separate from trained-policy or gray-specialization evidence.

## Versioned proposals and metadata

`graygo/reachable-multigroup-collision-probes/2`, selected with `--generator multicell`, first invokes the unchanged version-one proposal algorithm. If its first change leaves a zero-liberty group, that proposal is rejected. Otherwise it changes a second distinct occupied point by removal or recoloring, rejecting unstable final groups. If five points are already empty, the second change must be recoloring. Original holes stay empty, each accepted pair differs at exactly two points, and at most five empty points keep every complete root game within the original six-action budget, including pass.

The additional draws are part of the new generator's fixed stream. Version-one generation remains a separate wrapper around the shared exact construction core; its seed/proposal/certificate/trace behavior is preserved. Version-two catalogs include an explicit `controlled_changes` map from family IDs to the two changed positions. A new version label cannot relabel one-cell cases, omit controlled changes, reduce the 32-case count or promote repeated classes. Historical schema/identities remain readable unchanged.

Both boards are still reached from empty through 199 actual default-rule rounds using the general anchored setup builder. Actual full-state matrices are recertified after setup, and the complete base/remove game-pair identity must match the prototype. Gray final rings retain the actual extra placement that clears the padding suicide restriction before the last collision. No counters, histories or restrictions are assigned synthetically.

## Exact verification and reproducibility

The current independent auditor dispatches by the recorded generator version. It derives removal/recoloring counts from actual board differences, verifies explicit two-point metadata and repeats the original proposal prefix under its own declared ceiling. Its version-one path can still import the original frozen `7da67d9` package, while the preserved original auditor/receipt remain at their recorded hashes.

The combined suite passes **279 tests in 179.308 seconds**. All 31 probe tests pass. The fixed 500-proposal version-two prefix contains five distinct pairs (proposals 18, 54, 63, 138 and 354), four with mixed-equilibrium witnesses and one with gray roots. The tests reproduce every exact proof/trace, verify two changed points and stable groups, preserve the six-action bound, reject geometry copies, reject one-cell relabeling and preserve incomplete status. They also establish that this prefix adds complete-game classes beyond the audited version-one 24.

An additional direct check matches the version-one first 200 proposals' accepted proof/trace dictionaries byte-schema-for-byte-schema against the independently audited original (proposals 84 and 129). The existing source-bound one-cell construction/audit and historical geometric fixture remain unchanged. The bounded version-two development check took 8.315 process-wall seconds; tests/reproduction and broader development overhead are additional.

## Bounded construction

```sh
.venv/bin/graygo build-probes --generator multicell --seed 974215 --run-dir runs/collision-probe-multicell --attempts 200000 --time-cap-hours .5
```

A concrete clean-source allocation is recorded separately before launching a longer run. The search stops at 32 distinct certified classes or its explicit cap, retains partial proofs if incomplete and selects no training/scientific campaign. A full source-seed/exact-proof audit remains required before accepting completed coverage. The design's eight-hour CPU probe stage, original stopped costs and separately scoped overhead remain visible; these commands do not infer a full-campaign accelerator price.


## Complete frozen benchmark and independent audit

The separately allocated clean-source version-two run is now terminal and complete. It found **32 distinct complete-game families** in 33,066 proposals / **417.546 measured CPU seconds**, before its 1,800-second cap. The [tracked full catalog](probe-multicell-v2-construction/benchmark.json), [construction provenance](probe-multicell-v2-construction/README.md), [independent audit](probe-multicell-v2-audit/README.md) and [freeze receipt](../experiments/collision-benchmark-v2-freeze.json) are retained. The independent audit repeats all original proposals and validates every exact proof plus all 12,736 legal setup rounds in **421.945 seconds**, leaving the original inputs unchanged.

Coverage is balanced sixteen increasing/sixteen decreasing pairs, with 64 distinct public roots/current boards. Every pair changes exactly two positions; all complete legal games remain within six actions. Minimum certified range separation is **43/400 = 0.1075**, above 0.10. Twenty pairs contain gray roots and thirteen have mixed-equilibrium witnesses. Candidate-policy outputs were never used. The complete catalog is frozen at `artifacts/collision-benchmark-v2.json` before confirmatory policy inspection.

This completes structural benchmark coverage only. No learned policy is graded here. The all-three-seed 24/32 behavioral criterion, actual exposure-controlled performance/interaction and scientific conclusions remain pending. Oracle fixture passes cannot substitute for trained policies.
