# Current recorded control commitment and missing device prices

This is a source-referenced reconstruction of **21 recorded experimental table-control/profile leaves plus the selected live corrected R4 table reservation**. The old fifteen-leaf inventory, old completed R4 table, two corrected R1 prices, corrected R1 table and two corrected R4 prices are all retained. Failed and unused historical work is not discarded. Every original run manifest/result referenced by the old inventory is checksum-matched before building the example.

Actual recorded leaf intervals sum to **26991.557927068962898 CPU process-wall seconds**. The selected R4 table loop/setup reserve adds **15330 seconds**, giving **42321.557927068962898 seconds** under **43,200** and leaving **878.442072931037102 seconds**. The decimal sum differs from the earlier binary-float rollup by only **-1.127E-12 seconds**, retained in [the sources](sources.json); no negative adjustment or changed case time is invented.

The reservation covers the entire selected case and is not added to a partial live actual. Neural CPU, probe, audit/development and GPU measurements outside this control subset are not claimed as included. `project_consumption_complete` is false. The R4 process and allocation remain governed by their original source/caps; this report launches nothing.

The prospective device ledger is **uncosted**. P/U/B_U are explicitly zero device-hour quantities for the recorded CPU-only PPO/no-other-route-pilot/no-uniform scope, with references. C1/C4 device prices and H_P/H_D/H_N remain null; CPU toy timing is not converted to a GPU campaign forecast. The fixed known-term subtotal is a partial ceiling calculation, not measured consumption or a minimum spend. No project device ceiling or execution authority is supplied.

[Plan](plan.json), [usage inventory](usage.json), [immutable source references](sources.json) and [actual CLI report](report.json) are retained. The CLI can reproduce the report without private model inference; evidence hashes identify the underlying records, while independent full-runtime pricing qualification remains outside this calculator.

```sh
.venv/bin/python tools/nfsp_resources.py --plan docs/verification/nfsp-resource-ledger-current/plan.json \
  --usage docs/verification/nfsp-resource-ledger-current/usage.json
```
