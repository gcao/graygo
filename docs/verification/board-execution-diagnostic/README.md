# Native/CUDA board executor integration evidence

Clean measured revision: `0b15b8c0c6a773dfca72dd29ae5a8dc1218b41fd`.
Package SHA: `418c9dfc6031fb17cb88045c00aa18f28b1212a65948a51b8bb29863305a210f`.
Full retained run on `ai`: `/home/gcao/proj/graygo/runs/board-execution-R4-native-0b15b8c`.

All JSON reports, commands, frozen plans, final clocks, run/control/opening manifests and permanent public/final checkpoint manifests are copied byte-for-byte. `SHA256SUMS` indexes these JSON copies. Heavy networks and buffers remain at the retained run; the copied JSON files alone are not restorable model bundles.

The clean committed driver used the explicit checksummed native C++ decoder and full CUDA 64-channel/eight-block models. Two trajectories each reached 16,384 records under separate unchanged cumulative 300-second ceilings. The uninterrupted trajectory used one CLI process. The resumed trajectory used three: stop at 4,096; resume until the closed-reservoir opening pause at 8,192; then apply the explicit diagnostic opening and finish. All four commands and closed execution clocks remain visible.

The independent audit reloads sealed actual buffers and runtime, checks native/source/config/clock/run/control/public model bindings, complete archive/retention inventory, balanced own-role counts and absence of early labels. Exact complete saved state and all allocated buffer arrays match. Live CUDA optimizer-step device placement is checked separately by the executor tests, since this artifact comparison loads both checkpoints onto CPU.

Collection across the four CLI processes took 61.154 seconds. Persisted component clocks were 26.021 seconds for the full run and 32.286 seconds across all three resumed sessions, with zero unclean-recovery charge. They are a breakdown within the driver measurement; initialization before its timer and the independent audit are additional overhead. These are allocated process-wall observations, not kernel-active GPU time or campaign prices.

This remains `purpose=diagnostic`, with reduced opening/hand-off at 8,192/12,288, 32,768 aggregate records, no A2 completion and no formal scope approval. It establishes execution/recovery mechanics, not learned strength, a formal 5M opening decision or H_P/H_D/H_N. Formal entry/A2/resource/cohort evidence and the remaining design requirements are still pending.

Read-only re-audit using the measured package source:

```text
.venv/bin/python tools/check_board_execution.py runs/board-execution-R4-native-0b15b8c --audit-only
```
