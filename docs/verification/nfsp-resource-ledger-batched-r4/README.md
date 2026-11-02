# Selected batched R4 prices and completed-control settlement

This version extends the [earlier scalar-price subset](../nfsp-resource-ledger-measured-r4/README.md) with the [complete selected-source group profile and independent audit](../nfsp-batched-costs/README.md) and [completed, independently audited R4 table case](../nfsp-eligibility-R4-tabular/README.md). It does not discard failed historical work or claim complete project consumption.

The new B=32 / sixteen-pair prices are H_P **56.236005**, cumulative H_D **91.965357**, and H_N **254.932641** device-hours per seed. The source is package `f4c8b427d823d204df6bd46371b38b35bcb776d783ca04addd6831f84e666cbd`, selected one-step reuse four, native binary `f40906e...` and physical RTX 3080 Ti UUID `GPU-446c23a3-a50d-fb18-6e54-d0b290eb78c2`. H_D includes H_P, so no extra pilot prefix is added. The original corrected A2 cases remain on their separate frozen package; this record does not approve cross-source learning-gate reuse.

The measured pilot exceeds its initial **24-device-hour** component. Separate science/probe work is **24.819918** hours against **16**, an 8.819918-hour shortfall. The formula keeps the original 16-hour coefficient, and [scope review](scope-review.json) explicitly records the measured overrun. Known design-formula terms sum to **3,447.415814** hours; using measured science work gives a partial **3,456.235733**-hour forecast. Both omit unpriced C1/C4 device terms and unreconciled historical use. N_A2/full campaign totals remain null, no overall project device ceiling is declared, and no larger scope is selected.

## Disjoint control accounting

The earlier whole R4 table reserve of 15,330 seconds is removed. Its actual loop, measured pre-save setup and remaining unmeasured overhead receive three distinct coverage identities. Recorded control actuals now sum to **36,385.713606670905941 CPU process-wall seconds**; **28.584193724032957** remain reserved. Their commitment is **36,414.297800394938898** under **43,200**, leaving **6,785.702199605061102** unallocated. This retains unknown setup/finalization overhead rather than inventing zero use. The old tiny float/decimal rollup difference remains in the historical sources. No additional control or extension is funded automatically.

The new whole physical profile interval adds **0.2848548212 device-hours**. Its nested scalar-core interval is not added again. Together with the earlier recorded scalar profile, this declared GPU subset is **0.3280563443 device-hours**. The new independent price audit and completed-control audit add 7.289225 and 35.843425 CPU process-wall seconds under verification, not control. Other tests, experiments, neural CPU work and reports outside the inventory remain unclaimed. GPU memory sampling is not kernel-active time or a proof of the absolute device peak.

[Plan](plan.json), [usage](usage.json), [formula/usage report](report.json), [scope review](scope-review.json) and [source references](sources.json) preserve the exact inputs. Source-bound empirical accounting is independently checked; execution/A2/scope flags remain false.

```sh
.venv/bin/python tools/nfsp_resources.py --plan docs/verification/nfsp-resource-ledger-batched-r4/plan.json \
  --usage docs/verification/nfsp-resource-ledger-batched-r4/usage.json
```

The original parent continues the separately fresh R4 neural case at its unchanged 15M/15,300-second caps. Current control misses, missing cohorts, measured cost overruns and unpriced terms remain separate facts. A viable learner plus performance work or an explicit reduced scope is still required before a production launch.
