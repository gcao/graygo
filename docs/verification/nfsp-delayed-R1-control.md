# Independent full-budget delayed R1 control audit

The seed-0 tabular primary control completed all **15M own-side records**, split evenly by role. The read-only auditor uses pinned source `9fefeee`, SHA `ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e`. It derives values and strategy gaps analytically from terminal outcomes rather than invoking the trainer's task evaluator. The [audit](nfsp-delayed-R1-control/audit.json) binds the original run manifest, final checkpoint, episode-header checksum and exact auditor source.

| Final scheduled record | Worst average gap | Worst early Q error | Worst greedy-Q gap |
|---|---:|---:|---:|
| 13M | 0.631854 | 1.146153 | 0 |
| 13.5M | 0.599469 | 1.148212 | 0 |
| 14M | 0.558140 | 1.158365 | 0 |
| 14.5M | 0.550265 | 1.149998 | 0 |
| 15M | 0.501305 | 1.155510 | 0 |

The independently recomputed result is **not_ready**: greedy-Q root gaps pass, supported early-state Q values miss the 0.10 bound, and deployed average gaps miss the 0.10 readiness band. Average gaps are still improving at the cap. This is a complete-run grade at the declared budget, not proof of an implementation bug, a convergence theorem, or a license to move opening/change history.

The audit checks all 30 permanent scheduled model archives against their original-game values/gaps, own-policy versus native frozen-opponent continuation, both branch signs, horizons/controllers, and early/late/root-action values. It independently regenerates every episode's mode/ramp RNG and schedule, frozen generations, balanced 8/1/1 coverage, Algorithm-R selections, replay/supervised RNG continuations, forced-action RNG exclusions, optimizer/refresh clocks and per-role/origin totals. It checks all retained 500k executed transitions and 500k label observations against opaque public-state identity, true terminals, countdowns, source/opponent versions and legal masks.

Actual totals are 87,500 complete games, 12M ordinary records, 811,200 eligible labels, 742,023 accepted reservoir changes, 29,289 Q updates, 15,621 exact-average publications and 1,171 target refreshes. The table contains 2,788 public-role states. Seventeen have potentially overwritten historical labels in the final unpublished sub-batch: the audit reverses the Algorithm-R tail and checks legal probability-mass bounds for those rows. The other **2,771 published means, including all 12 root means, are checked exactly** against retained targets. It does not claim to reconstruct the missing old targets for those 17 rows, every historical greedy target, or full action trajectories from header-only completed-game logs. Replay/label path unit tests and checksummed originals remain separate evidence.

The new regression checks exercise both tabular and neural completed diagnostics, including partial final episodes and read-only file preservation. Deliberately resealed wrong native-opponent values and private-RNG drift are rejected. The combined suite passes **181 tests in 67.000 seconds**.

Reproduce from the project root on `ai`:

```sh
PYTHONPATH=/home/gcao/proj/graygo/.worktrees/nfsp-tasks-9fefeee/src .venv/bin/python tools/audit_nfsp_delayed.py runs/nfsp-delayed-R1-primary-9fefeee/tabular --output runs/nfsp-delayed-R1-primary-9fefeee/tabular/audit-independent.json
```

The original driver PID 2217027 is still live in the matching neural case and must finish its own full budget or explicit time cap. No R4 selection, milestone extension or 9x9 learning approval is made by this audit. The remaining seeds, short/bounded/branching checks, production controller, resource and scientific work retain their full scope.
