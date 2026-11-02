# Measured R4 prices and unchanged scope limits

This version extends the [previous declared subset](../nfsp-resource-ledger-current/README.md) with the [independently verified selected-R4 measurement](../nfsp-r4-full-cost/README.md). It retains all 21 historical control leaves and the entire live R4 table reserve, without adding a partial live actual to that reservation. Original control commitment remains 42,321.557927 CPU process-wall seconds under 43,200.

The source-bound empirical device prices are now H_P **52.744889**, cumulative H_D **89.165897**, and H_N **257.485589** per seed. They refer only to learner package `db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873`, one-step reuse four, the recorded native binary and RTX 3080 Ti UUID. A source/target/reuse/hardware change needs its own compatibility/pricing evidence. H_D includes H_P, so the formula adds three H_D without another three H_P. C1/C4 device prices remain missing; CPU table/neural profiles are not silently GPU prices. N_A2 and the full campaign total remain null.

The measured pilot exceeds its initial **24-device-hour** per-seed limit. Separately, the empirical scientific/probe forecast is **76.616486 device-hours** against the design component of **16**. The formula calculator preserves the original design coefficient; [scope-review.json](scope-review.json) explicitly records its 60.616486-hour shortfall. No scope/funding increase is made. The known-term design-formula subtotal is 3,469.367963 device-hours; replacing that underfunded component with its measured forecast gives a partial 3,529.984449-hour candidate subtotal. Both omit unpriced A2 terms and unreconciled historical consumption; neither is a full project price or permission to spend.

The inventory adds only the original measurement's **155.5254831-second physical GPU interval** (0.0432015231 device-hours) and the independent artifact audit's **6.620969 CPU process-wall seconds**. Epoch-interval and monotonic elapsed measurements remain separately retained in the original job; the physical ledger uses the interval endpoints exactly. There is no concurrency amortization or restarted measurement. This declared subset is not complete project consumption.

[Plan](plan.json), [usage](usage.json), [formula/usage report](report.json), [scope review](scope-review.json), and [source references](sources.json). All execution/A2/scope flags remain false. The mathematical formula and physical interval union can be reproduced without retraining:

```sh
.venv/bin/python tools/nfsp_resources.py --plan docs/verification/nfsp-resource-ledger-measured-r4/plan.json \
  --usage docs/verification/nfsp-resource-ledger-measured-r4/usage.json
```

The companion scope review additionally uses the independently audited scientific cost rather than pretending the fixed design coefficient is a measured price. Optimization with a newly recorded selected-source measurement, or an explicit reduced-scope decision, is required before a feasible production launch. The ongoing original R4 A2 case remains unchanged and still requires its full endpoint and independent learning audit.
