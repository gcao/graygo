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
