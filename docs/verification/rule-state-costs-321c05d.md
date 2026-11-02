# Compact rule-state reference costs

Source commit `321c05d1ec2b924a302ef9e7483444812c734c0a`, SHA `b6fa3b686f2d39cb54e5dcfd8073e9ec84a6d97923cd4258925502fcd9a09e57`. Identical legal 9×9 trajectories, seed 42, produced 20,000 records per storage representation. Capacity was 500,000 per buffer, so this is a partial-fill IO/decode benchmark, not full-capacity resident memory or learner throughput.

| Storage | Collection seconds | Replay decode seconds | Reservoir decode seconds | Checkpoint seconds | Strict restore seconds | Snapshot payload bytes |
|---|---:|---:|---:|---:|---:|---:|
| features | 9.126 | 0.040 | 0.005 | 0.502 | 0.084 | 2932834 |
| rule-state | 5.818 | 7.396 | 3.702 | 0.173 | 16.360 | 824284 |

Decode timings cover 20 batches of 512 starts each; replay regenerates both source and successor features. All-legal tie labels are synthetic storage data, not learned Q targets. Both representations passed verified save/load and decoded-sample round trips. The original paired profiler completed in 43.328 CPU process-wall seconds, concurrent with the existing first screens.

The compact representation saves logical host arrays and compressed payload/collection cost, but the initial Python decoder is much slower. These timings must not be used to claim production readiness or price a full campaign. A focused profile identifies repeated numeric-ABC validation of already canonical integer colors as the leading overhead, followed by feature/group generation. The raw profile is preserved beside the report.

A tested integer fast path now avoids those generic checks for ordinary Python colors while retaining Integral normalization, boolean/range rejection and all resolved-state/history invariants. This implementation change must be remeasured on the exact same snapshots before deciding whether native/batched decoding is still required. Game rules, feature contents, labels and loss/update schedules are unchanged.
