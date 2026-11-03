# Audited graph prices and declared resource use

This version extends the [prepared-resolver inventory](../nfsp-resource-ledger-prepared-r4/README.md), preserving every prior leaf and the original control settlement. The [complete graph measurement and independent audit](../nfsp-graph-costs/full-cost/README.md) provide H_P **18.852895**, cumulative H_D **38.176705**, H_N **131.821295**, and separate science/probes **14.923266** device-hours for package `a522fe2...`, one-step R4, selected graph execution, B=32/sixteen-pair evaluation and one RTX 3080 Ti.

Both measured initial components now fit without changing the design: pilot headroom is **5.147105** hours below 24, and science/probe headroom **1.076734** below 16. This resolves the earlier component overruns for this later package. C1/C4 prices, N_A2/full campaign total, compatible complete A2 evidence and a project ceiling remain missing. The known design-formula subtotal is **1,808.932490** hours; replacing its fixed science component with the measured forecast gives a partial **1,807.855756** hours. Neither is a complete campaign price or a selected scope.

The 39 declared leaves include the original 77.8917558-second matched CLI diagnostic, 1.6874719-second failed graph preflight and 318.1608319-second corrected whole graph profile. Combined known physical allocation is **0.546835685139 device-hours**. Nested profile phases, cold children and the original paired learner parent are not added again. The original R4 neural loop contributes 9,671.735913 CPU process-wall seconds; its fresh audit and graph-price audit contribute distinct 33.085825 and 13.463699 verification seconds. This is still incomplete project consumption: other tests, prototypes, historical neural/setup/report costs and CPU work remain outside this declared subset.

Control accounting is unchanged: **36,385.713606670905941 actual + 28.584193724032957 reserved = 36,414.297800394938898 seconds**, under 43,200. The **6,785.702199605061102** unallocated seconds select no new control, seed or extension. Unknown original table overhead remains reserved even though the parent is terminal.

[Plan](plan.json), [usage](usage.json), [source inventory](sources.json), [formula/usage report](report.json) and [scope review](scope-review.json) retain exact quantities and input references. Empirical component accounting is verified; runtime qualification, A2 exit and execution authority are separate and remain false.

```sh
PYTHONPATH=src:. .venv/bin/python -m tools.nfsp_resources \
  --plan docs/verification/nfsp-resource-ledger-graph-r4/plan.json \
  --usage docs/verification/nfsp-resource-ledger-graph-r4/usage.json
```
