# Original full-budget delayed R4 neural result and independent audit

Original driver PID **2320416** is terminal/absent. Both matched seed-0 R4 cases completed their unchanged 15M-record allocations. The neural case took **9,611.440 CPU process-wall seconds**, with 7.5M fresh own-side records per role. Frozen source `9fefeee` / SHA `ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e` is unchanged from its launch; no current eligibility repair was applied to the live comparison. The [audit](nfsp-delayed-R4-neural/audit.json), [complete result](nfsp-delayed-R4-neural/result.json), [manifest](nfsp-delayed-R4-neural/run.json), [funding profile](nfsp-delayed-R4-neural/budget.json) and [frozen paired plan](nfsp-delayed-R4-neural/plan.json) retain the exact evidence.

| Final scheduled record | Worst average gap | Worst early Q error | Worst greedy-Q gap |
|---|---:|---:|---:|
| 13M | 0.000002897 | 0.245661 | 0 |
| 13.5M | 0.000001237 | 0.200983 | 0 |
| 14M | 0.000001005 | 0.263632 | 0 |
| 14.5M | 0.000000389 | 0.260394 | 0 |
| 15M | 0.000000318 | 0.222295 | 0 |

The final-five neural-average numerical maximum is **0.000002897**, below 0.10; greedy-Q root gaps are zero. Supported native Q value errors have final-five maximum **0.263632**, above 0.10. The composite readiness remains `not_ready` because credit misses. **This is not a neural averaging failure.** It also does not qualify for the fixed marginal extension, whose last point must pass and earlier misses must stay within twenty percent of threshold.

The independent auditor checks all thirty original scheduled model archives against closed-form original-game gaps and native-opponent values, full episode/mode/ramp/coverage/Algorithm-R/private-RNG accounting, optimizer/refresh clocks and all retained replay/label rows. Actual counts are 87,500 complete games, 12M ordinary records, 811,200 eligible labels, 117,156 Q updates / 59,983,872 Q targets, 15,621 supervised updates / 7,997,952 targets and 1,171 target refreshes. All 500k retained executed records and 500k label observations are checked. Header-only logs do not reconstruct every completed action trajectory or every historical greedy label; the audit retains that scope.

The independently reproduced [supplemental-opponent defect](nfsp-forced-opponent-bias.md) remains a distinct interpretation issue. Original passive forced-root samples do not have the native opponent kernel and yield a stationary bias near 0.20 in the primary example. The preserved R1/R4 credit misses therefore cannot cleanly select against one-step propagation or justify moving straight to Retrace. Other observed long-horizon approximation failures are not explained away. Correct per-role value eligibility and record clocks, revalidate independent audits, then profile and record any corrective scope before new long training. All original thresholds remain unchanged.

The paired original R4 result is complete diagnostic evidence only. No remaining R4 cohort, stability extension, target comparison, later opening, 9x9 pilot, protected promotion or scientific study is selected by this report.

Reproduce the historical audit using its original package and auditor:

```sh
PYTHONPATH=/home/gcao/proj/graygo/.worktrees/nfsp-tasks-9fefeee/src .venv/bin/python runs/nfsp-delayed-R4-primary-9fefeee/audit_nfsp_delayed.py runs/nfsp-delayed-R4-primary-9fefeee/neural --budget-profile runs/nfsp-delayed-R4-primary-9fefeee/neural-budget.json --output /tmp/graygo-R4-neural-audit.json
```
