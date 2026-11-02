# Real-board NFSP runtime verification

## Implemented component

The collector advances independent full-rule 9x9 environments while keeping each player's logical mode, actual acting policy, epsilon, private action RNG and model version frozen through its episode. Public inference is grouped by immutable snapshot and acting policy. Both commitments for every selected environment are completed before any of those environments resolves. Mode tags still control frozen greedy-label eligibility; exploratory executed actions remain the Q-replay actions.

The runtime combines the shared NFSP learner, compact rule-state buffers, optional explicit native decoder, independent Q and averaging sample/augmentation RNGs, fixed logical batch/update/refresh clocks, target inference, optimizer work and atomic recovery. Each sampled replay start uses one D4 symmetry plus torus translation for its current/next observations, full sequence context, action, legal masks and frozen targets. Pass, rewards, logged probabilities and global features stay invariant.

Checkpoints retain all active public states, episode logs/choices, old frozen snapshots, environment cursor, per-environment/private/global RNGs, replay/reservoir arrays and insertion clocks, optimizers/targets, Torch/CUDA RNGs, deterministic CUDA settings and collector metrics. Restore binds the source, runtime configuration, start distribution and exact native binary/schema. CUDA workspace configuration is established before initializing the context.

## Verification

The board-runtime tests cover scalar/batched private sampling parity, resumed partial environments with old policy versions, exact CPU continuation with augmentation, exact CUDA continuation on 9x9, exact CUDA continuation using the full 64-channel/eight-block models, geometric transition/label correspondence, sequence padding/context preservation, and rejection of unbounded hardware-profile endpoints. Exact continuation compares Q/target/average weights, raw buffers, metrics, counters and RNG state with uninterrupted execution at zero tolerance. These are same-hardware continuation checks, not a claim of bit-identical CPU/GPU arithmetic or cross-platform portability.

The full previous suite plus the first six new checks passed 176 tests in 61.904 seconds. The final eight board-runtime checks passed in 3.595 seconds, including both CUDA cases. The final combined suite passed all **178 tests in 62.210 seconds**.

## Hardware profile and remaining integration

`profile-board` declares both phases before collecting data. It fixes full 9x9/200-round gray rules, 64 channels/eight blocks, logical batch 512, learning start 4,096 and refresh 12,800. The closed phase retains production opening settings. A separately initialized early-opening diagnostic exercises supervised and average work with explicitly different opening endpoints, without pretending to have reached 5M/7M records. Each phase is limited to 32,768 records and the combined ceiling is at most one hour. Signals/time limits are checked between collection chunks and complete paired games; atomic checkpoint IO and a complete color pair may overrun the check boundary. It cannot issue pilot/A2/promotion approval.

Reports retain full configurations, source/environment/native identity, actual record and per-role counts, optimizer/refresh/target counts, collection/learning/setup/IO/evaluation time, peak CUDA allocation/reservation and process-lifetime RSS, endpoint checkpoint inventory and coverage flags. Both Q and average are timed against the frozen random/tactical baselines with one complete color pair each. Those tiny matches are timing diagnostics, without confidence or strength claims. A changed time cap, model or source is a new declared measurement, not an implicit continuation.

This remains a low-level collector/learner component. It does not yet provide the A2-exit controller, fixed pre-/post-opening checkpoint journal, development label-opening approval, all-seed milestone decisions or independent confirmation/scientific orchestration. Full-capacity native buffer IO has separate verified evidence; a short untrained profile cannot replace trained-phase cost/memory measurements or fill the full campaign ledger by itself.


## Completed full-model GPU hardware measurement

Source commit `dc97531` (`47ed7a7b188eb883d35e7fb03e04e6d55635d2c612cb48795b759ee72a26104c`) completed the declared measurement on `home-ai`'s RTX 3080 Ti. Original profile PID 2271075 is terminal; both its final report and process termination were checked. The run took **50.966 seconds** total, below its fixed 720-second ceiling. It used 16 environments, full 64-channel/eight-block models, logical batch 512, R1, deterministic CUDA, paired augmentation and the checksummed native rule decoder. Each phase retained both full 500k-capacity arrays (203 MB combined), with only the actual 16,384 fresh records populated.

| Actual component | Closed | Early-open diagnostic |
|---|---:|---:|
| Fresh own-side records | 16,384 | 16,384 |
| Q updates / targets | 25 / 12,800 | 25 / 12,800 |
| Supervised updates / targets | 0 / 0 | 26 / 13,312 |
| Target refreshes | 1 | 1 |
| Eligible reservoir labels | 0 | 3,028 |
| Collection seconds | 15.415 | 17.523 |
| Learning seconds | 2.280 | 3.035 |
| Checkpoint write / strict restore seconds | 0.434 / 0.467 | 0.407 / 0.420 |
| Eight paired-color evaluation games, seconds | 4.860 | 4.768 |
| Total phase seconds | 24.441 | 26.330 |
| Peak CUDA allocated bytes | 814,530,560 | 830,899,200 |
| Peak CUDA reserved bytes | 1,042,284,544 | 1,008,730,112 |
| Process-lifetime peak RSS bytes | 2,078,449,664 | 2,160,947,200 |
| Actual checkpoint payload bytes | 48,880,981 | 43,887,602 |

Both phases passed every component-coverage check. The early-open phase recorded 9,971 average-policy and 6,413 Q actions; its 3,028 logical best-response visits exactly equal eligible labels. The closed phase recorded no labels, SL updates or average acting. Each phase's 25 Q updates follow the 4,096-record warm-up; the configured R1 cadence is not an assertion that every warm-up record had a loss. All endpoints, restored clocks, raw ordinals, buffer capacities, source/native identities, complete episode accounting, scored evaluation totals and payload inventories passed the separate [read-only audit](../../tools/audit_board_profile.py). Run it with the measured source on `PYTHONPATH`.

The [plan](board-runtime-profile/plan.json), [environment manifest](board-runtime-profile/run.json), [complete measurements](board-runtime-profile/report.json) and [audit hashes/checks](board-runtime-profile/audit.json) are archived. Checkpoint bundles remain at the run path on `ai`. The report's pre-save `timings.checkpoint` is zero; the explicit checkpoint write/restore fields above are the actual measured IO. RSS is a process high-water mark, so the second phase includes the first phase's history.

This is **not** an 80M-record price: untrained self-play averaged short games and frequently mutual-passed, the diagnostic changed opening endpoints, and the full buffers were allocated without being filled here. The separate full-capacity IO test supplies filled-buffer evidence. Trained-phase game distributions, production milestone/cohort integration, official evaluation sample budgets and the applicable A2/resource decisions remain required before filling H_P/H_D/H_N or launching development.
