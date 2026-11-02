# Strategically distinct reachable collision-probe construction

## Versioned candidate-independent proposals

The original liberty generator remains available as `--generator liberty`; its 32 geometric regression pairs still cover one complete-game class and cannot pass family coverage. The new `--generator multigroup` uses version `graygo/reachable-multigroup-collision-probes/1`. Both formats retain their exact generator identity, and seed regeneration dispatches to the recorded constructor. Historical catalog identities and proof contents are unchanged.

The new generator grows two/three-faction regions from one to four permanent liberties, with a fixed black/white balancing weight, gray seeding/expansion rate and hole-count stratification. Every grown component has a path to a permanent empty point. It then removes or recolors exactly one stone, rejecting zero-liberty groups. A rare unseeded intersection isolated by four permanent holes can start its own group at that direct liberty. No learner, candidate output or opponent commitment is inspected.

Before expensive setup, complete one-round utilities receive an exact rational near-equilibrium screen. Utility-constant games are safely rejected because both pass and a shared placement are legal, admitting collision rates zero and one. Other games use the existing complete vertex certificate at individual gap 1/20, with all legal actions and pass retained. The separation test uses exact 1/10 arithmetic. Failed/budget-exceeded certificates do not qualify a case.

## Complete-game uniqueness and balance

Each accepted pair must have a different complete base/remove payoff-pair identity under shared placement relabeling, global color exchange and pair reversal. Current-board geometric/color copies and reused roots are rejected too. Reversing the same pair never creates another family. Accepted unique classes are oriented in a fixed alternating increase/decrease schedule, yielding sixteen of each only when all 32 unique pairs are present.

This retains the existing explicit operational family-equivalence test, with no binomial population interpretation. It prevents thirty-two geometry variants or generator labels from manufacturing family coverage. Partial catalogs preserve their actual qualified class count and proofs while remaining incomplete. No target count, threshold or trained-seed acceptance rule is relaxed.

## Actual default-game setup

The general builder reaches each dense multigroup board from the official empty state by actual resolver calls. It orders each color from permanent liberties and prioritizes a shared protected ring. Gray stones are created by collisions, not assigned to an unreachable board or inserted via counter edits.

After the ring is surrounded, one opponent suicide and at most one simultaneous pass can pad each intervening real board change. Forbidden moves clear only through those legal board changes. Black/white final rings use suicide/pass/friendly-fill for the last three rounds. A gray final ring requires both players, so it reserves an additional real placement for round 198 to clear the padding restriction before the round-199 collision. Passing alone would retain that restriction and is rejected by the trace tests. Anchored components remain live throughout construction.

Both roots are required to match their full 199-round traces, 9x9/200-round rules, counters, histories and restrictions. Their complete actual-root certificates are regenerated after setup. A setup that changes the prototype's complete game pair fails. The constructor never assigns a fake round number, resets a history or substitutes a short-rule prototype into the benchmark.

## Feasibility and verification scope

The [frozen preliminary search](probe-diversity-feasibility/README.md) completed 9,382 proposals in 180.021 CPU process-wall seconds and found eleven distinct prototype classes, including mixed equilibria and gray-containing roots. These include the original motif, rather than eleven guaranteed additional families. The current trace tests replay all 22 default-game roots, 4,378 legal rounds, and independently verify every exact certificate without invoking the floating LP solver during proof reading.

Additional tests cover three final-ring factions, per-player suicide restrictions and a real restriction-clearing gray placement, invalid/sparse setup rejection, deterministic removal/recolor proposals, full source-seed reproduction of a small partial catalog, repeated geometric class rejection, incomplete/interrupted/time-capped construction and historical catalog compatibility. A complete production catalog still requires thirty-two distinct certified pairs and its terminal source/seed/lineage audit. Exact equilibrium teacher fixtures do not establish a trained agent's behavior or gray specialization.

## Bounded construction and verification

```sh
.venv/bin/graygo build-probes --generator multigroup --seed 974214 --run-dir runs/collision-probe-diverse-v1 --attempts 200000 --time-cap-hours .5
.venv/bin/graygo verify-probes --benchmark runs/collision-probe-diverse-v1/benchmark.json --regenerate --time-cap-hours .5
```

These commands illustrate a separately recorded construction allocation within the design's eight-hour CPU probe ceiling. Check the original OS process and run outputs before restarting anything; partial work and failed proposals remain in the report. If thirty-two families cannot be certified within the allocation, report incomplete coverage. This constructor starts no learning or confirmatory inference campaign. Resource accounting and actual collected results are recorded separately after terminal verification.


## Test result

The combined suite passes **253 tests in 116.002 seconds**. All 24 probe-specific certification/catalog/diversity tests pass, including historical byte/schema compatibility, the eleven complete-game prototype classes, legal 199-round setup traces, gray restriction clearing, exact certificate reading, seeded partial reconstruction and duplicate-class rejection. The small new-constructor check yields two distinct classes at proposals 84 and 129 of its fixed 200-attempt prefix; it remains correctly incomplete.
