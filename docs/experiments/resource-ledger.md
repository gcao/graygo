# Recorded experiment resources

These are completed sequential **CPU process wall times**, not accelerator-hours or a complete development-cost total. Raw state artifacts on `ai` supply each value. Tests, independent audit runtime and small engine/synthetic-network profiles are additional overhead not rolled into this table; do not call its sum complete project consumption.

| Completed component | Seconds | CPU process wall hours |
|---|---:|---:|
| PPO matrix preflight | 1082.360 | 0.30066 |
| NFSP terminal-matrix timing | 38.253 | 0.01063 |
| Asymmetric NFSP first screen | 2776.246 | 0.77118 |
| Historical delayed timing | 232.936 | 0.06470 |
| Corrected delayed timing | 229.264 | 0.06368 |
| RPS NFSP first screen (both completed cases) | 2782.993 | 0.77305 |
| Delayed R1 first screen (completed tabular case only) | 7218.411 | 2.00511 |

The historical delayed cost is retained even though its source cannot fund the corrected evaluator. The corrected cost report enters C1 and the separate tabular-control ceiling; C4 remains conditional. RPS is complete and independently audited. The delayed tabular case completed the full 15M records and reports a gate miss; its independent task audit remains pending. Its matching neural case is still live and is not counted as a completed allocation. R1 delayed first-screen trial settings are bound to the measured budget hashes before launch. H_P, H_D, H_N, CE and the full-study envelope remain unfilled until their applicable selection/profiling steps. CPU terminal/opaque-task costs never substitute for 9×9 accelerator measurements.


## Bounded accelerator pipeline measurement

The full-model 9x9 R1 hardware exercise (`dc97531`, `verification/board-runtime.md`) completed two 16,384-record phases in **50.966 seconds**, or **0.01416 elapsed hours of a GPU-equipped process**. This is an allocated-process wall time, not measured kernel-active GPU time. It includes collection/learner setup, full-network Q/target/averaging work, strict checkpoint write/restore, and 16 small timing-only evaluation games. The CUDA allocation peak was 830,899,200 bytes; reservation peaked at 1,042,284,544 bytes. The original profile process is terminal, with audited final evidence.

The early-open phase is an explicit diagnostic. Untrained game lengths, partial buffer populations and tiny evaluation samples do not establish H_P/H_D/H_N. The independent full-capacity storage profile and previous CPU experiments remain separate measured costs; tests/audits/builds/development overhead are still not comprehensively summed. No large campaign allocation follows from this measurement.
