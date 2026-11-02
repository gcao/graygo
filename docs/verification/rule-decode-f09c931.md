# Integer-fast-path decoder remeasurement

Source commit `f09c9316a74c81f23bca1a0dca820f8a4064e81f`. This read-only follow-up uses the exact original 20,000-record feature/rule snapshots, verified by their manifests/checksums, with fresh matched sampler streams. The profiler confirms identical decoded records and frozen targets across representations.

| Storage | Strict restore seconds | Replay decode seconds | Reservoir decode seconds |
|---|---:|---:|---:|
| features | 0.087 | 0.016 | 0.005 |
| rule-state | 6.516 | 4.059 | 2.019 |

Sampling uses 20 batches of 512 starts. Sampler seeds are preserved in the plan; comparison with the earlier timing is over the same corpus rather than identical random start indices. Concurrent RPS/delayed training and CPU/cache variability preclude an isolated speedup claim.

The integer fast path materially lowers the measured compact-decoder/validation cost, but rule-state sampling is still much slower than direct feature unpacking. Logical memory remains 203 MB for the paired 500k buffers. Do not use this microbenchmark to price 9×9 learning: native/batched decoding, full-capacity IO and the complete learner profile remain necessary. The target has an available Ubuntu C++ 13.3 compiler for a separately verified acceleration backend; no native backend is claimed here.

The original snapshots and timing remain intact. No learner history, rule, feature, reward, label or acceptance threshold changed.
