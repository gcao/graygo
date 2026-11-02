# Corrected R1 seed-0 neural screen, complete and audited

The separately fresh neural case completed all **15M aggregate records** in **7717.382732 CPU process-wall seconds**, below the unchanged 15,300-second ceiling. Original paired driver PID **2444001** is terminal/absent and its durable state is complete with two trials. No original run was restarted. Frozen learner source remains `ca45a53` / `db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873`.

The original pinned independent auditor verifies the full case through the source-bound A2 collector. Audit driver PID **2532352**, case-audit PID **2532353**, assessment PID **2532520** are terminal, verified. Driver cost is **32.252770 measured CPU seconds**, with every original case input unchanged. It independently recomputes all thirty model archives, native opponent-policy values, coverage, private streams, Algorithm-R/replay/label/update/refresh clocks and retained buffers.

| Final evaluation records | Max average gap | Max greedy-Q gap | Max supported Q error |
|---:|---:|---:|---:|
| 13,000,000 | 0.000011204 | 0.000000 | 0.410784 |
| 13,500,000 | 0.000011082 | 0.000000 | 0.321584 |
| 14,000,000 | 0.000005512 | 0.000000 | 0.210173 |
| 14,500,000 | 0.000003167 | 0.000000 | 0.166339 |
| 15,000,000 | 0.000002247 | 0.000000 | 0.120614 |

Worst final-five deployed-average gap is **0.000011204**, within 0.10; greedy-Q gap is **0**, within 0.05. Supported error is **0.410784**, above 0.10, so delayed credit misses. The final error improves to **0.120614**, but the fixed final-five window is unchanged. This is **not a neural averaging failure**. It also cannot receive the marginal extension: the final value does not pass 0.10 and earlier excursions exceed 0.12. No later opening, entropy or checkpoint sweep is selected.

The audit checks **14,982,500 eligible-Q records**, with **17,500 passive forced-opponent root records excluded only from that denominator**; all 15M records remain budgeted. The exact source/audit/result/final-checkpoint hashes are retained in [the snapshot](snapshot.json), along with the [original manifest](run.json), [result](result.json), [audit](audit.json), [durable command record](audit-job.json), [case snapshot](case.json), [funding](budget.json), [exact auditor](audit_nfsp_delayed.py) and [driver](audit_driver.py).

The [paired assessment](paired-assessment.json) retains two completed seed-0 cases and **34 missing A2 slots**. Table credit and averaging miss; neural average passes numerically while credit misses. Neither case completes A2, and no 9x9 learning/promotion/confirmation/scientific claim is authorized. Completed historical bias-confounded source/results stay preserved separately.

Portable archived reanalysis uses both retained snapshots:

```sh
.venv/bin/python tools/a2_cohort.py assess --plan docs/verification/nfsp-eligibility-R1-neural/cohort-plan.json \
  --case docs/verification/nfsp-eligibility-R1-neural/tabular-case.json \
  --case docs/verification/nfsp-eligibility-R1-neural/case.json --archived
```
