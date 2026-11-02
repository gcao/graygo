# Exact collision-probe certification and coverage

## Complete one-round proof

At a complete public root with one round remaining, the matrix contains **every** selectable black/white action pair, including pass. Each leaf uses the actual simultaneous resolver and exact terminal area-score utility. The root value is its own value; arbitrary asymmetric roots are never assigned the empty-game value zero.

The floating matrix solver proposes rational equilibrium witnesses. Exact fraction arithmetic requires nonnegative normalized strategies whose lower and upper security bounds coincide exactly. Verification reads the supplied witnesses, proves these equalities and rebuilds the complete rational bounds without invoking the floating solver.

For exact value `v` and `e=1/20`, enumerate the vertices of:

```text
P = {p >= 0, sum(p)=1, p^T A[:,j] >= v-e for every column j}
Q = {q >= 0, sum(q)=1, A[i,:] q <= v+e for every row i}
```

Every independent active constraint basis is solved by rational Gaussian elimination; candidates are checked against all inequalities. Singular/duplicate bases are handled exactly. Collision probability is bilinear, `sum_{a != pass} p[a]*q[a]`. Its minimum and maximum over the compact product `P x Q` occur at vertex pairs: holding one argument fixed leaves a linear objective in the other. Thus exhaustive vertex-product evaluation supplies global bounds over the whole near-equilibrium strategy sets. Resource exhaustion yields no certificate; there is no heuristic truncation.

Proof artifacts include exact payoff tables, complete action indices, value/equilibrium witnesses, both vertex inventories, collision extrema and witnesses, budgets, rules and public roots. A paired case qualifies only when its two complete ranges are separated by at least `1/10`; the preferred direction is fixed from that proof. Policy grading uses exact original-game gaps for both roles at both roots and the required `1/20` directional collision shift.

## Reachable default-game regression instances

The version-1 constructor uses one controlled liberty-change motif. Independently generated connected black/white regions retain a shared liberty. A minority stone is added or removed to change the number of empty liberty points from one to two. Region geometry, color leadership, location and orientation vary; all current boards and complete public states are checked modulo torus/D4/color symmetries, with no root reuse.

The setup builder actually plays from the empty default board. It grows connected groups while retaining permanent liberties, uses a protected friendly ring for legal isolated-opponent suicides, and interleaves single mutual-pass rounds with board-changing placements. Each root has a replayable **199-round** trace, the default 200-round limit, correct two-entry public history, cleared restrictions and zero mutual-pass streak. No clock or history is fabricated. Both paired histories differ only at the controlled stone, with identical recorded moves. The same reference resolver checks every prefix transition, including the no-change restriction/clearing rules.

All 32 pairs have an actual collision-rule utility effect: the base/ablation payoff tables differ on a placement-collision cell. Their individual near-equilibrium bounds satisfy the original 0.05/0.10 requirements. The largest-collision root has range `[361/400, 1]`, the smallest-collision root `[0, 1/20]`, giving separation `341/400 = 0.8525`.

## Independence and acceptance remain incomplete

The design explicitly excludes symmetry copies. This implementation additionally uses a conservative strategic-equivalence check: complete base/remove payoff pairs and collision incidence are canonicalized under common placement-label permutations, global color exchange and pair reversal. Identical complete strategic problems on different board geometries count as one family. This is an explicit coverage policy, not a theorem of statistical independence.

The current 32 geometric pairs reduce to **one strategic family**. They are regression variants of that motif, not a completed 32-family behavioral benchmark. The catalog and verifier report `incomplete_family_coverage`; `benchmark_complete` stays false. A repeated family passes only if every included instance passes. Raw instance counts remain diagnostic. Missing seeds/cases, duplicated symmetry cases, edited bounds, changed thresholds, invalid setup traces and incomplete proof budgets cannot become a pass.

The acceptance machinery retains the full requirement: at least 24 of 32 independent families in **each** of seeds 200, 201 and 202, both roles/roots within 0.05, and collision shift at least 0.05. It emits no binomial population confidence bound and cannot claim gray specialization from a solver fixture. Additional strategically distinct probe families and actual frozen trained-policy evaluation remain required. Mixed-strategy cases will broaden coverage beyond the current pure-strategy motif.

## Frozen-policy semantics and verification

Neural public inference shares the existing action-sampling path. An initial marginal uses the same normalization without drawing an action. Episode mixtures take the weighted marginal of independently sampled private components. Each measured probe episode starts at the declared root; the 199-round construction trace is a reachability certificate. The default continuation API rejects a fresh mixture prior after an acting-policy prefix; that case needs history-conditioned component weights from the strategy evaluator.

The tests cover exact analytical mixed/nonuniform/degenerate matrices, excluded pass collisions, full finite best responses, all vertex/witness checks, full-state geometric identity, budget failure, edited proofs, forged setup traces, family/seed coverage, the 24-instance boundary without false family acceptance, private-mixture marginals and unchanged RNG state. Native/reference encoding is compared for every root and every terminal leaf: **480 complete states / 960 perspective observations**. The final combined suite passed **204 tests in 76.211 seconds**.

Construction and solver-fixture evidence is separate from A2, the production milestone/controller, 9x9 campaign cost, trained-policy provenance and the exposure-controlled performance/interaction tests. All remain part of the full objective.


## Source-bound construction record

Clean source commit `2fdf1df` (`2095459225127c84f9c5d4f5b48cf5deefb316949851ba85e7d01b0b9253e471`) generated the catalog in 51 proposals and 4.915 seconds under a prewritten 180-second plan. Its logical SHA-256 is `8be1da3f84db16686b921383456eb09e32729c7ac297f27b830b6cf27ce77d3a`; the exact generated file SHA-256 is `644fb75d809ff4aa90ead6f0ffe6ba2af8a2f89b2af6d8b085db93430ca3a787`. It is byte-for-byte identical to the versioned [regression fixture](../../tests/fixtures/collision-benchmark-v1.json), avoiding a duplicate large payload. The [construction plan](../experiments/collision-probe-catalog-v1/plan.json), [environment/source record](../experiments/collision-probe-catalog-v1/run.json), [incomplete-coverage result](../experiments/collision-probe-catalog-v1/report.json) and preserved [diagnostic disposition](../experiments/collision-probe-catalog-v1/diagnostics.json) accompany it. Raw run artifacts remain on `ai`.

All 64 roots have legal default-game setup traces (12,736 replayed rounds), and all 32 geometric pairs have a collision-rule payoff effect. Independent reconstruction reproduces the exact catalog, and exact-equilibrium solver fixtures pass all instances. **There is still only one strategic family**, so these observations cannot clear the 32-family acceptance requirement.

The earlier bounded diagnostics are retained: the 3x3 recoloring search produced five geometric pairs in two strategic classes; the 5x5 dense search produced 32 pairs but one strategic class and no collision-rule utility effect; liberty-change searches produced rule-sensitive instances of the same pure-strategy motif. The dense diagnostic includes roots requiring both roles to mix, which is useful direction for a broader constructor. None is represented as completed independent-family coverage or trained-policy evidence.


The final clean-source [verification](../experiments/collision-probe-catalog-v1/verification.json), commit `3f56fd9` / source `068e550570d9d1bf7ca9a524103ec82ade6a60745cebe9cc6a81c5984e35b015`, reproduced the catalog in 5.053 seconds. It explicitly separates **32 rule-sensitive geometric pairs from one rule-sensitive strategic family**, checks 64 distinct boards/public roots, replays 12,736 setup rounds and verifies all exact solver fixtures. Its status is `incomplete_family_coverage`. No trained-policy or scientific-performance result is implied.
