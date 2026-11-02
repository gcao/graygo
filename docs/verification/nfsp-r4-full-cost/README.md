# Selected R4 full-component cost measurement

The original CUDA measurement is terminal and successful: driver PID **2560327**, profiler PID **2560331**, frozen producer commit `0c0228a`, learner commit `ca45a53` / package `db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873`. The producer and original driver remain retained without rerunning them. The [recorded allocation](../../experiments/nfsp-r4-cost-measurement-allocation.json) authorized only this bounded measurement.

## Observed result

| Quantity | Measurement / candidate forecast |
|---|---:|
| Original GPU-equipped driver elapsed | 155.525482 seconds |
| Physical GPU allocation interval | 155.5254831 seconds = 0.0432015231 device-hours |
| Actual new diagnostic records | 65,792 |
| Offline full-corpus rows | 500,000; zero new environment actions |
| Full replay + supervised arrays | 203,000,000 bytes |
| Peak CUDA allocated / reserved | 835,935,232 / 1,012,924,416 bytes |
| Process lifetime peak RSS | 2,851,037,184 bytes |
| Full sealed IO payload | 119,470,797 bytes |
| Full write / strict restore | 3.423183 / 3.078041 seconds |
| H_P: 8M pilot per seed | 52.744889 device-hours |
| H_D: 20M cumulative development per seed | 89.165897 device-hours |
| H_N: 80M confirmation per seed | 257.485589 device-hours |
| Separate scientific evaluation / probes | 76.616486 device-hours |

These forecasts use two times the largest observed component cost, selected reuse four, one allocated process/device, full buffer sampling, sixteen potentially distinct frozen episode versions, the declared checkpoint cadence and maximum 200-round evaluation games. The forecast evaluation scopes are 36k games at 8M, 40k cumulative at 20M, and 44k at 80M; the separate scientific scope prices 96k games. They are assumptions for costing, not game allocations. `H_D` includes `H_P`; do not add the pilot prefix twice. CPU control use is separate.

The **pilot forecast exceeds the initial 24-hour per-seed pilot ceiling**, and the **scientific/probe forecast exceeds its 16-hour component**. The original scope is therefore not affordable at these empirical ceilings. The private version-diverse actors and evaluation are the largest components. No larger training scope follows from this result; optimization and another recorded selected-source measurement, or an explicit reduced-scope decision, remain necessary. Complete A2 and whole-project consumption remain unresolved.

## Independent audit procedure

`tools/audit_nfsp_cost.py` never imports the producer. It rebuilds all rates from raw timing samples, subtracts disjoint nested actor/copy costs, checks actual pipeline optimizer/fresh/target clocks and paired games, and independently counts every 8M/20M/80M event. It verifies the original source/tool/allocation/report bindings, physical interval and caps, actual checkpoint seals, full buffer schemas and retained transitions, offline executed-action labels, strict full model layouts, both saved optimizers, sixteen separate frozen snapshot storages, independent maximum-length logs and public-load payload. It holds a shared read lock, hashes retained inputs again at the end, and writes to a new output directory outside the original run. It does not reproduce physical timings or claim a runtime theorem.

The source package, producer and allocation fingerprints must be supplied independently by the caller. The full-model CUDA profile is required: CPU/reduced/changed-schedule measurements cannot become production prices. All outputs retain `selected_scope_qualified=false`, `runtime_bound_proven=false`, `a2_complete=false` and `execution_authorized=false`.

Thirteen targeted tests pass on the retained real measurement and adversarial undercount/omission/mismatch cases. The existing complete suite remains the previously recorded 376 tests; no new full-suite run is claimed. The actual artifact audit is complete: frozen auditor commit `c24f7ed`, SHA `36c737a86337efc5a85368e95a62154f75d944d895681bd5d45c674507826ecc`, 6.620969 CPU process-wall seconds. It verifies all retained original inputs unchanged and reproduces both cost-limit violations. The [audit output](audit.json) SHA is `741533c3c60a357612fd8ca87de53238d7575cf5b6415a401040dbf1d0d52fa6`. This verifies source-bound empirical accounting; it does not qualify the initial scope.

## Retained files

[Report](report.json), [pipeline](pipeline.json), [plan](plan.json), [source/hardware manifest](run.json), [original OS job](job.json), [driver](driver.py), and [targeted tests](auditor-tests.txt). Large sealed buffers and timing-only model payloads remain in the ignored original run; they are not deployed agents or learning evidence.

The [updated partial ledger](../nfsp-resource-ledger-measured-r4/README.md) enters these measured H values, keeps C1/C4 device prices missing, retains the original control reservation and adds the actual physical profile interval and independent audit. Its companion scope review separately compares measured science/probe work with the unchanged 16-hour design component; it does not silently replace that ceiling.
