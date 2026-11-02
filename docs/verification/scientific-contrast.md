# Matched scientific contrast implementation

## Fixed pairing and game allocation

The protocol freezes all twelve training-arm/seed policies: training starts `{empty, exposure}` x training rules `{gray, remove}` x confirmation seeds `{200,201,202}`. Within each training-start condition, the base-trained policy P_G plays the variant-trained P_E under **both** evaluation rules. Evaluation uses official empty roots plus the shared versioned held-out-gray distribution. Color roles exchange on the same complete root, so an opening's color advantage is not preserved for one agent.

One independently seeded block contains:

```text
2 curricula x 2 opening contexts x 3 frozen training seeds
x 2 evaluation rules x 2 agent colors = 48 physical games
```

The confirmation looks are 500 and 2,000 blocks: 24,000 and 96,000 total games, including held-out diagnostics. Each block shares the same held-out family/orientation across rules and policies, and the same private policy seeds across evaluation rules. Both actions still use the same pre-reveal public state. All 48 games commit in one SQLite transaction; an interrupted/failed block contributes no partial rule, seed or context. Repeated trajectories remain observations.

This is an explicit allocation for the specified four-arm scientific contrasts and held-out diagnostics. Broader exploratory cross-play and the protected promotion round retain their separately declared plans/budgets. The master scope ledger must account for all selected evaluation and probe-inference work; the API does not select the large campaign.

## Endpoints and conditional inference

For each seed/context/curriculum, retain paired-color u_G, u_E and their within-block difference. The primary endpoint is official-empty/exposure. Its two equal-seed-weighted aggregate lower bounds must be strictly above +0.05 for u_G and u_G-u_E, and the corresponding point estimates must be positive in **every** training seed. Secondary official-empty/empty results cannot rescue a failed primary effect. Held-out-gray results remain diagnostics.

The confidence unit is the complete matched block. For aggregate endpoints, first average the three seed values within a block, then compute the existing scaled empirical-Bernstein lower bound across blocks. This retains rule/opening/seed dependence within the unit. Utilities use support [-1,1]; interactions use [-2,2]. Forty-eight predeclared aggregate/seed endpoints over two looks share alpha 0.05 (delta=0.05/96). The source method remains `confidence.py`'s referenced finite-sample theorem.

These are **conditional game-sampling bounds for the fixed registered cohort**. Three training seeds are reported individually; thousands of games do not create a training-seed population interval. No binomial population interpretation is applied to probes. A numerical effect alone never declares gray specialization: the report explicitly leaves A2, training-scope approval, actual exposure matching and independent probe coverage unverified.

## Provenance, recovery and resource discipline

Confirmation collection checks permanent sealed archives against the actual frozen policy weights, seed, training rule, 80M checkpoint, confirmation run scope and exact training-start identity. All twelve run manifests must be distinct. Diagnostics may use baselines, but cannot be relabeled as confirmation models. Checkpoint metadata checks are separate from independent validation of approved training scope and completed A2/probe/exposure prerequisites.

The immutable plan binds the protocol, held-out catalog, all policies, source version and numerical runtime (device/hardware, dependency versions, thread count, deterministic/TF32/workspace settings). Resume rejects a changed registry, catalog, source, ceiling or runtime. The twenty-GiB reserve applies to actual evidence writes. Setup, collection, in-run validation and commits consume a persistent checksummed clock; reporting is separate CPU analysis overhead. An unclean in-flight interval is conservatively charged including offline time. Normal interruption saves the observed cumulative time without charging the pause. This prevents repeated invocations from refreshing a collection ceiling; atomic setup/block/IO may overrun a check boundary.

Scored payloads, contract and clock have separate integrity hashes and exact prefix checks. SQL block insertion and clock-prefix advancement commit together. Read-only analysis opens the database read-only, reconstructs every opening/seed/component draw, checks all score totals/signs and full cell inventories, and refuses to hide a later completed look.

## Evaluation-version correction

The shared paired evaluator is now **version 2**. Previously it swapped both board colors and agent colors for the second game, preserving an asymmetric opening advantage for the same agent. On a legal board with three black stones and one white stone, identical pass policies returned paired utilities [+1,+1]. Version 2 exchanges the agents on the same public opening, producing [+1,-1] and mean zero. The regression verifies the unchanged board score in both role assignments.

The pairing operation is unchanged on official-empty roots. Historical empty-root observations retain their original interpretation and identities; a new source revision still records its own policy IDs/seeds. Version-1 nonempty-opening evidence cannot be interpreted as version-2 color-balanced evidence; old source-bound artifacts remain available under their original revisions, and new collection uses the new contract. Resume and protected-evidence readers reject a changed evaluation version rather than silently mixing semantics.

## Verification scope

The new tests cover exact physical/block budgets, simultaneous coverage and support, paired interaction variance, conditional equal-seed aggregation, a bad seed despite positive aggregate bounds, secondary-only success, missing seeds/rules/contexts, hidden later looks, malformed pair values, scored/root tampering, exact interrupted continuation, partial-block and transaction rollback, clock corruption/unclean restart, immutable ceilings/runtimes, read-only analysis and rejection of synthetic development containers as confirmation inputs. The combined suite passes **218 tests in 77.843 seconds**. CLI diagnostic evidence is recorded separately after collection.

This implementation launches no scientific training. The selected learner's complete A2 gates, production milestone/cohort controller, numerical campaign resource decision, twelve independent confirmation runs, matched actual curriculum consumption and the full independent probe benchmark remain required work.


## Completed CLI integration diagnostic

Clean source `74b7275` / SHA `8293daffd44af5219208647e14896e64fe6a154453fd0e51c2c2b3649727835c` completed a predeclared small random-baseline exercise. One run collected the two-block/96-game look and resumed to four blocks/192 games; a separate uninterrupted run collected 192 games. All scored blocks and analysis receipts match exactly, and the read-only CLI reports the same evidence hash. This is **384 actual games**, not 384 training runs or independent trained replicas.

The [audit](scientific-contrast-diagnostic/audit.json), [frozen plan](scientific-contrast-diagnostic/plan.json), [source/environment](scientific-contrast-diagnostic/run.json), [clock/progress](scientific-contrast-diagnostic/progress.json), [summary](scientific-contrast-diagnostic/summary.json), [baseline registry](scientific-contrast-diagnostic/registry.json) and complete [SQLite evidence](scientific-contrast-diagnostic/evidence.sqlite) are retained. The copied database/contract are byte/data checked by the read-only reader. The resumed collector charged 6.211 seconds of observed setup/collection and zero recovery gap. Process startup/reporting and the uninterrupted collector are separate overhead; no full-study runtime forecast is inferred.

Both records remain `purpose=diagnostic`, the primary numerical gate does not pass, and `gray_specialization_demonstrated=false`. Arm/seed labels in this exercise are evaluator wiring labels for random baselines. They are not confirmation training replicas, A2 evidence, calibrated power evidence, matched-training proof or an empirical four-arm performance result. No confirmation training was launched.
