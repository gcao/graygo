# Complete-component one-step NFSP cost profile

`tools/nfsp_cost_profile.py` measures the selected full 9x9 learner with full sampling buffers, private actor-version stress, actual kernel/augmentation paths, collection/IO and evaluation costs. It produces a **candidate forecast**, not a learned-policy result, runtime theorem, independently qualified price or permission to train a pilot/study. Existing small pipeline/storage measurements remain separately scoped.

## No invented training history

The input is a sealed full rule-state replay corpus generated previously by legal games. Retained valid transitions are sorted and materialized into a new offline one-step dataset. Its record indices explicitly belong to that materialization, not an original optimizer's fresh clock. Valid retained links are reindexed for realistic IO data; only one-step kernels are timed, and no Retrace price follows. Known retained Q eligibility is counted for the new dataset without inferring an unknown legacy history. Synthetic legal executed-action targets fill its supervised sampling corpus and are never claimed as NFSP greedy history.

Production `update_q`, one-step online-policy/frozen-target inference, clipping, gradient/Adam updates, `update_average`, paired geometric augmentation and actual full-buffer sampling are timed directly. Real nonterminal starts stress the complete bootstrap path without changing any transition or done flag. Warm-up and measured optimizer steps/targets are explicit. These kernel updates do not fabricate NFSP fresh-record counters and cannot create a valid runtime resume/public agent archive.

New diagnostic collection uses the real private collector, resolver, packet writes, Algorithm-R offers and metrics. Small adapters expose a local new-record cursor while mapping appended rows and stream IDs into distinct full-buffer namespaces. This offsets actual collected records; it does not set nominal 5M/80M counters or claim to restore an old collector. Dense timing roots are the independently generated complete public round-199 states, with the real official rules and legal next transitions. Their one-round endings stress resolver/terminal/storage overhead; they are declared timing roots, not a training curriculum.

## Measured components

The source's real bounded closed/post pipeline runs with synchronization/timing wrappers. The wrappers do not alter commitments, updates or clocks. Actor and snapshot-copy intervals are removed from its collection total before deriving the resolver/storage/bookkeeping rate. Dense full-buffer overwrite measurements supply another real collection rate; the larger observed rate is used.

Private actor stress measures shared Q, shared average and distinct Q/average snapshot groups. Both roles see the same complete public state. Sixteen immutable episode versions can produce thirty-two independent inference groups, addressing variable game lengths that a synchronized untrained collector may not exhibit. Source snapshot copy cost is priced once per possible optimizer publication boundary, not once per historical context transition or once for every replay draw.

Full sealed IO includes both 500k buffers, the three learner networks, both optimizers, the declared number of frozen episode models and maximum-length log metadata. Strict model/buffer round trips are checked. Its files use cost-only names rather than valid public `models.pt` or runtime `runtime.pt` artifacts. Model loading and parameter-integrity checks are also measured.

Actual paired evaluations cover Q/average versus random/tactical and average versus average. Empty/dense legal round calls supplement whole games so a short untrained game cannot hide per-round cost. No strength or confidence claim follows from these timing games.

## Forecast assumptions

The forecast keeps logical batch 512, selected Q reuse, 4,096-record warm-up and target copy every 12,800 fresh Q records. For official board collection all records are Q eligible. It uses the exact possible Q-update count, a conservative supervised count after the fixed 5M opening, snapshot publications, checkpoint cadence plus milestones, public archives, restore/setup and declared evaluation scopes. H_D is cumulative and includes the pilot; it does not add H_P again. Default per-seed evaluation budgets are 36k pilot games, 40k cumulative screening and 44k confirmation diagnostic games, with the 200-round cap explicitly priced. These are recorded pricing assumptions, not an implicit allocation or permission for those games.

Maxima of measured component samples and a stated runtime margin produce per-seed 8M/20M/80M ceilings. The separate 96k-game scientific/probe calculation includes both-neural evaluation, probe forward calls, model loads and integrity work. Concurrency is one allocated process/device; eight-job amortization is not assumed. CPU results remain process-hour projections and yield null device-hour H_P/H_D/H_N. The forecast remains empirical: a future trained distribution or hardware contention can exceed observed maxima, and operational caps/scope decisions remain mandatory.

`candidate_price_forecast_available` additionally requires the full model/board/cadence, full 500k buffers, CUDA/native decoding, adequate source pipeline/kernel/actor/dense measurement counts and deterministic augmentation. Reduced diagnostics cannot qualify. Independent artifact/accounting review and source/hardware scope approval remain separate from this flag.

## Command

```sh
PYTHONPATH=SELECTED_FROZEN_LEARNER/src .venv/bin/python tools/nfsp_cost_profile.py \
  --expected-source-sha PACKAGE_SHA --reuse 4 --device cuda \
  --buffer-checkpoint SEALED_FULL_BUFFER_CHECKPOINT \
  --benchmark docs/verification/probe-multicell-v2-construction/benchmark.json \
  --native-library artifacts/native/BUILD/libgraygo_rule_decode.so \
  --run-dir runs/selected-cost-profile --time-cap-seconds 3600
```

The one-hour maximum is an original invocation ceiling, not a renewed campaign allocation. An existing profile cannot be overwritten. Actual hardware experiments require their separately recorded source/limit allocation; a proposed forecast never opens an A2, pilot, confirmation or study gate.

## Validation scope

Tests exercise real small-board materialization, nonterminal full-buffer kernel work, private paired/versioned commitments, full-buffer append offsets, strict model/buffer IO, CPU/device unit separation and exact reuse/refresh/pilot-prefix counts. A complete reduced diagnostic drives real source runtime instrumentation and all remaining measurement stages; it cannot produce a CUDA campaign price. These are profiling correctness checks, not convergence evidence. The live corrected R4 trial remains on its original frozen source and original process.

The full suite passes **376 tests in 272.402 seconds**; all ten affected profiler tests pass again after final retained-link/independent-log IO fidelity changes. [Full](nfsp-cost-profile-tests.txt), [targeted](nfsp-cost-profile-targeted-tests.txt) and [validation/source](nfsp-cost-profile-validation.json) records are retained. A separately [allocated selected-R4 CUDA run](../experiments/nfsp-r4-cost-measurement-allocation.json) can now collect actual full-model evidence; tests do not supply a measured price.

## Actual selected-R4 measurement

The original allocated full CUDA measurement completed successfully in 155.525482 driver seconds. Its [retained candidate forecast](nfsp-r4-full-cost/README.md) is H_P 52.744889, H_D 89.165897, H_N 257.485589 and separate science/probe 76.616486 device-hours. The first and last exceed the initial 24-hour and 16-hour components. Independent artifact/accounting review is provided by the separate read-only `tools/audit_nfsp_cost.py`; its thirteen adversarial tests pass. Neither a successful measurement nor a verified forecast selects an affordable production scope.
