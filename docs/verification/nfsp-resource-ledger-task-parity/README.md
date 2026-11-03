# Declared resource use after task source parity checks

This version preserves all 39 leaves in the [graph-price ledger](../nfsp-resource-ledger-graph-r4/README.md), adding the [trained actor-boundary verification](../task-source-compatibility-ca45-c3b8/actor-boundary/README.md), eight matched diagnostic task prefixes, original strict-checker cost and [corrected read-only comparison](../task-source-compatibility-ca45-c3b8/pipeline/README.md). There are 54 actual/reserved measurement leaves. The mixed whole driver is not added to its child/case intervals.

Control use is now **36,551.066872905881251 actual + 148.11725165514508252 reserved = 36,699.18412456102633352 seconds**, below 43,200. The four new table cases contribute **165.353266234975310** measured loop/known-setup seconds. Their still-unknown overhead retains the residual of each predeclared thirty-second setup allowance; the earlier 28.584194-second reserve is preserved. The original new 840-second commitment settles to 284.886324 seconds. **6,500.81587543897366648** seconds remain unallocated; this does not select another experiment or waive an A2/control requirement.

CPU neural recorded use is 9,842.609080 seconds and verification 519.190510 seconds in this partial inventory. The actor driver is counted once (119.571101 seconds), alongside the original 1.618802-second strict comparison and corrected 3.387266-second comparison. Child actor intervals and the whole mixed profiling parent are not double counted. Other tests, infrastructure, historical CPU work and unmeasured neural/wrapper costs remain outside the declared subset; no complete project-consumption claim follows.

No new accelerator work occurs. Known physical allocation remains **0.546835685139 device-hours**. The independently audited source-bound graph forecasts are unchanged: H_P **18.852895**, cumulative H_D **38.176705**, H_N **131.821295**, and science/probes **14.923266** hours. The measured 24/16-hour component limits still pass. C1/C4 prices, N_A2/full total and project ceiling remain missing. CPU diagnostic timing is not substituted for those prices.

[Plan](plan.json), [usage](usage.json), [source inventory](sources.json), [report](report.json), and [scope review](scope-review.json) preserve units and references. Finite source parity is verified; complete source compatibility, A2 evidence reuse, runtime qualification and execution authority remain separate and false.

```sh
PYTHONPATH=src:. .venv/bin/python -m tools.nfsp_resources \
  --plan docs/verification/nfsp-resource-ledger-task-parity/plan.json \
  --usage docs/verification/nfsp-resource-ledger-task-parity/usage.json
```
