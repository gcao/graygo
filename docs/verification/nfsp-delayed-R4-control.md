# Independent full-budget delayed R4 control audit

The seed-0 tabular primary control completed all **15M aggregate fresh own-side records**, 7.5M per role, in **9,250.354 CPU process-wall seconds**. It started from its separately verified zero state, with the same frozen `9fefeee` learner source as R1 (SHA `ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e`). The only learner change is Q reuse one to four; opening/ramp, exploration, coverage, target-refresh and evaluation budgets remain matched. The [audit](nfsp-delayed-R4-control/audit.json), [run](nfsp-delayed-R4-control/run.json), [complete result](nfsp-delayed-R4-control/result.json), [budget](nfsp-delayed-R4-control/budget.json) and [frozen plan](nfsp-delayed-R4-control/plan.json) bind the exact evidence.

| Final scheduled record | Worst average gap | Worst early Q error | Worst greedy-Q gap |
|---|---:|---:|---:|
| 13M | 0.353488 | 1.192802 | 0 |
| 13.5M | 0.339450 | 1.201866 | 0 |
| 14M | 0.313364 | 1.206570 | 0 |
| 14.5M | 0.281818 | 1.211164 | 0 |
| 15M | 0.266667 | 1.214686 | 0 |

The independently recomputed control grade is **not_ready**. Greedy-Q root gaps pass, but supported early-state values miss the 0.10 bound by a large margin. The final-five average maximum is 0.353488, above the 0.10 readiness and 0.25 lag bands; the final 15M average gap is 0.266667. Average gaps improve through the cap, while supported credit still misses. This is a complete-run grade, not proof of a neural implementation bug or a license to shift opening, discard history, replace the seed or relax thresholds. The final point does not pass the fixed requirements, so this case does not qualify for the marginal stability extension.

All thirty permanent scheduled model archives are checked by independent closed-form policy/value grades. The auditor regenerates episode modes/ramp choices, balanced 8/1/1 coverage, frozen generations, private RNGs, Algorithm-R decisions, optimizer/refresh clocks, true-terminal indexing and retained opaque replay/label observations. It checks 500k retained executed and 500k retained label records. There are 87,500 complete games, 12M ordinary records, 811,200 eligible labels, 742,023 accepted reservoir changes, 117,156 Q updates / 59,983,872 Q targets, 15,621 exact-average publications and 1,171 unchanged fresh-record target refreshes.

Published tabular means are exact for 2,771 public-role states, including all twelve roots. Seventeen final unpublished-tail states receive the documented legal-mass bounds because overwritten old labels cannot all be reconstructed. Completed full trajectories and every historical greedy label are not recoverable from header-only logs; the audit makes no such claim. This is the same explicit audit scope as R1.

The [reservation settlement](../experiments/nfsp-delayed-R4-tabular-settlement.json) replaces the 18,900-second control reservation with its measured 9,250.354 seconds. Recorded experimental control/profile time is now 19,409.090 seconds (5.39141 hours), leaving 23,790.910 seconds unallocated within the twelve-hour control ceiling. Tests, audits and bootstrap are additional overhead. This settlement selects no new trials.

The original driver PID 2320416 has moved to the matching neural seed-0 case. It retains its original full 15M budget/time ceiling; no early neural checkpoint substitutes for completion. The remaining seeds, possible target diagnostics, A2 cohorts and formal board learning remain unselected pending the applicable evidence/resource decisions.

Reproduce the audit from the project root on `ai`:

```sh
PYTHONPATH=/home/gcao/proj/graygo/.worktrees/nfsp-tasks-9fefeee/src .venv/bin/python runs/nfsp-delayed-R4-primary-9fefeee/audit_nfsp_delayed.py runs/nfsp-delayed-R4-primary-9fefeee/tabular --budget-profile runs/nfsp-delayed-R4-primary-9fefeee/tabular-budget.json --output /tmp/graygo-R4-control-audit.json
```
