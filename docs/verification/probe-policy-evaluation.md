# Frozen-policy collision benchmark evaluation

The complete candidate-independent version-two catalog now has a persistent evaluator for actual deployed policies. Construction and exact reachability/proof verification remain in [the benchmark report](probe-multicell.md); its [freeze receipt](../experiments/collision-benchmark-v2-freeze.json) is an input, alongside the exact catalog file. This component collects behavior and verifies archive bindings. Actual learned confirmation behavior and the exposure-controlled performance study remain pending.

## Fixed policies and units

`ProbePolicyProtocol` binds the benchmark, complete freeze receipt, original seeds 200/201/202, deployment representation, curriculum, policy identities and one cumulative inference ceiling. Confirmation takes the selected `empty/gray` or `exposure/gray` arm directly from the already frozen twelve-policy scientific registry. It requires average policies, distinct original seed identities, the same exposure catalog and the fixed 80M confirmation milestone.

The archive verifier reloads the actual permanent weights and checks rules, seed, scope, milestone, full 64-channel/eight-block model, primary replay/batch/target and optimizer settings, a valid complete confirmation journal plan, deterministic augmentation, start distribution, training-manifest checksum and journal-plan seal. The scientific evaluator shares these checks. Archive metadata alone does not verify an actual training trajectory, resource authority or matched exposure history; the report keeps those prerequisites unverified.

Each of the 32 transaction units contains **both roots, both roles and all three original policies**: 384 full action marginals in total. Each root starts a new probe episode, with its complete round-199 state and legal actions. The public setup traces establish legal reachability; they are not an opponent-policy observation prefix. Exact initial marginals use the same mask/normalization as deployed private sampling. Independent episode mixtures require their proper initial marginal semantics. No sampled action, oracle action or restricted candidate action set replaces a deployed marginal.

The grader uses the unchanged exact complete-game certificates: individual gap at most 1/20, collision shift at least 1/20 in the certified direction, and at least **24 distinct passing families for every original seed**. The benchmark has 32 complete-game classes. This is a fixed finite benchmark, without a binomial population confidence claim.

## Persistence, inference audit and budget

SQLite stores the immutable plan and one checksummed row per complete family. A failed inference or SQL transaction commits no subset. Resume binds source, models, benchmark, policies, numerical runtime and the original ceiling. A nonblocking OS lock serializes writers. Reports read the original database without writing progress or requiring a model process. Output guards protect the evidence directory, inputs and original training archives.

The sealed cumulative clock includes measured setup, inference, validation, commit and frozen-forward reproduction across all invocations. Unclean recovery conservatively charges the unresolved wall interval; clock reversal consumes the remaining reservation. Neither resume nor a separate audit command renews the ceiling. Units finish atomically, so the final report also checks actual measured consumption against the cap. Confirmation preserves the twenty-GiB disk reserve. Explicit diagnostics cannot allocate more than one cumulative hour; confirmation inference cannot exceed the existing sixteen-hour evaluation ceiling.

The default completed collection repeats all 384 frozen forward passes under the pinned source/model/runtime. A receipt binds the exact evidence prefix and model contract. Resealed legal marginals can be parsed, but they cannot pass forward reproduction if they differ from the actual deployed model. A separate `probe-audit` invocation consumes the same persistent budget and reloads the originally recorded archive paths. An incomplete prefix cannot pass the behavioral criterion. Numerical grades, archive bindings, forward reproduction and budget compliance are reported separately.

## Commands

```sh
.venv/bin/graygo probe-evaluate \
  --benchmark artifacts/collision-benchmark-v2.json \
  --freeze-receipt docs/experiments/collision-benchmark-v2-freeze.json \
  --science-protocol runs/scientific-study/plan.json \
  --catalog artifacts/gray-exposure-v1.json \
  --policy 200=RUN200/weights/000080000000 \
  --policy 201=RUN201/weights/000080000000 \
  --policy 202=RUN202/weights/000080000000 \
  --name confirmation-exposure --curriculum exposure \
  --time-cap-hours 1 --run-dir runs/confirmation-exposure-probes

# Resume the identical collection command with --resume.
.venv/bin/graygo probe-report --run-dir runs/confirmation-exposure-probes
.venv/bin/graygo probe-audit --run-dir runs/confirmation-exposure-probes
```

These are future confirmation paths, not existing trained weights or an allocation. A diagnostic explicitly selects `--purpose diagnostic`, binds original public archives for all three seed labels, and omits the scientific registry/catalog when unnecessary. Diagnostic or solver-fixture passes never satisfy confirmation. `--families` permits bounded transactional segments; `--no-audit` defers forward reproduction, which remains required for the confirmation behavioral flag.

## Verification and limits

The complete suite passes **301 tests in 213.847 seconds**, including thirteen new evaluator integration tests. They exercise actual untrained neural archives and real CLI collection/resume/report/audit, the complete 32-family exact-solver fixture, same-budget audit exhaustion, source/model/runtime/ceiling changes, missing seeds/roots/roles, illegal or invalid marginals, partial inference and SQL rollback, disk reserve, concurrent writers, clock corruption/recovery/reversal and resealed marginal tampering. The exact-solver fixture passes 32/32 numerically for each seed and reproduces its own forwards, while every confirmation/specialization flag remains false. Zero-record diagnostic weights are rejected as confirmation weights before evidence creation.

No training or real confirmation policy is supplied by these tests. A future behavioral pass must additionally satisfy the separate A2, training provenance/scope, exposure matching and scientific performance/interaction requirements before any gray-specialization conclusion. This evaluator never grants those prerequisites.

## Actual full-shape CLI verification

The separately allocated [zero-record diagnostic](probe-policy-cli-diagnostic/README.md) is terminal and verified from frozen `f0751b0`, with unchanged tested package SHA `9dbe715f543b2067a078693c943254010602616defaf5e3cecf81fe23ab5e10f`. Three actual full-shape untrained archives complete 8→32-family CLI recovery and a separately reloaded 384-marginal forward audit. The driver takes **34.723550 seconds**, with **23.148002 seconds** charged to its original 600-second cumulative inference ceiling. All three initial Q/average networks independently reconstruct from their seeds, the copied evidence reproduces read-only, and inputs remain unchanged. The untrained grade is **0/32 per seed**, with every confirmation/specialization flag false. The driver-only preflight Path error is preserved separately; no learning/inference occurred in that failed attempt. See the settlement for separately measured audit and additional overhead.
