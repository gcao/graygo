# Corrected matched R4 complete-pipeline pricing

The original pricing process **PID 2533933** is terminal and complete. Four separately fresh 65,536-record cases use the same corrected source as R1: `ca45a53` / package `db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873`. Q reuse is four; the 12,800-eligible-Q target clock, both-role horizons/coverage, opaque codebooks and all other nominal learner settings stay fixed.

| Timing case | Seconds including setup | Eligible Q records | Q updates | Supervised updates |
|---|---:|---:|---:|---:|
| tabular-R4-pre | 43.672724 | 65,461 | 480 | 0 |
| tabular-R4-post | 41.022945 | 65,461 | 480 | 86 |
| neural-R4-pre | 43.702084 | 65,461 | 480 | 0 |
| neural-R4-post | 40.246234 | 65,461 | 480 | 86 |

Every case has 65,536 aggregate records, 65,461 eligible-Q records, 245,760 Q loss targets, 75 excluded passive forced roots and five target refreshes. Pre-opening timing exercises closed-reservoir Q collection; the deliberately reduced post-opening timing exercises supervised learning and actual average acting. Reduced timing schedules cannot satisfy a formal learning gate. Buffers declare the full 500k capacities but are partially populated; no 9x9 or Retrace price follows.

The original independent audit driver **PID 2535532** is terminal/verified. All four exact pinned-source audits pass, including all private mode/ramp/action/replay/reservoir clocks, actual masks, retained storage and archived finite-game values. Original case inputs remain unchanged. Its driver took **9.002059 CPU seconds**, separate from **169.155873 seconds** for the pricing driver (**168.643987 seconds** summed measured case intervals, including **84.695669 seconds** for the two table leaves). Tests/development/import/archiving outside those intervals remain additional.

Phase-weighted full-15M forecasts are **9,591.569036 seconds** for the table and **9,475.292397 seconds** for neural. The original profiler's twofold proposals of **19,800 seconds** remain untouched. Before these measurements, the pricing allocation declared a separate possible funding review with a 15,300-second cap and minimum margin **1.5**, using the completed R1 timing calibration. Actual cap/forecast margins are **1.595151 / 1.614726**. This is a financial margin; all record caps, fixed windows and original-game thresholds are unchanged.

The [profile settlement](../../experiments/nfsp-eligibility-R4-profile-settlement.json) enters twenty-one control/profile leaves / **26,991.557927 CPU seconds**. The [separate full-pair allocation](../../experiments/nfsp-eligibility-R4-primary-allocation.json) reserves one table case plus 30 seconds of setup, projecting **42,321.557927 seconds** under the 43,200-second control ceiling. No further seed/control/target/campaign is funded. The [raw twofold budgets](tabular-raw-twofold-budget.json), [reviewed funding](tabular-reviewed-budget.json), [measured results](profile.json), [audit job](audit-job.json) and [all input copy checksums](snapshot.json) remain independently reviewable.
