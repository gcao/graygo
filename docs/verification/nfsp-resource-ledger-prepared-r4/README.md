# Exact tactical optimization: measured prices and declared use

This version extends the [previous declared subset](../nfsp-resource-ledger-batched-r4/README.md) with the [complete optimized-source price audit](../prepared-placement/full-cost/README.md). It retains historical controls/profiles and the completed R4 control settlement without relabeling prior source results.

Source `61c271fb94bd421b47aeb941fb43e03dcac307fd11a488720c0d560710d10eed`, one-step R4, B=32/sixteen-pair evaluation and the same physical RTX 3080 Ti yield empirical H_P **28.860612**, cumulative H_D **62.611484**, H_N **228.194172**, and separate science/probes **24.940015** device-hours. H_D includes H_P. The original 24/16-hour components remain unchanged and are still exceeded. [Scope review](scope-review.json) records both shortfalls; C1/C4 device prices, N_A2/full campaign total and project ceiling remain missing. The known design-formula subtotal is 3,038.600873 hours; using measured science work gives a partial 3,047.540888-hour forecast. These exclude unpriced work and unreconciled history and select no campaign.

The added **389.8655674-second** physical interval is counted once. Combined known profile allocation is **0.43635233525 device-hours**, not total project consumption. The nested scalar core and GPU cold child share that same physical interval and are not added again. The exhaustive/matched CPU verification, complete-game comparison and new independent price audit have distinct measured CPU scopes. Full-suite CUDA tests and other unlisted work remain outside this declared subset.

Control accounting stays **36,385.713606670905941 actual + 28.584193724032957 reserved = 36,414.297800394938898 seconds**, under 43,200. Its **6,785.702199605061102** unallocated seconds grant no new control/extension. The separately fresh original neural case continues unchanged. Physical-memory samples and tensor high-water marks are separately labeled; neither is a runtime bound.

[Plan](plan.json), [usage](usage.json), [source inventory](sources.json), [formula/usage report](report.json) and [scope review](scope-review.json) retain exact inputs. Empirical accounting is independently checked; no A2, scope or execution authority follows.

```sh
.venv/bin/python tools/nfsp_resources.py --plan docs/verification/nfsp-resource-ledger-prepared-r4/plan.json \
  --usage docs/verification/nfsp-resource-ledger-prepared-r4/usage.json
```
