# Corrected-source complete delayed costs

Source commit `9fefeeea1a343fe5d36f3cb42991b78dd335b001`, SHA `ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e`. All eight sequential CPU profiles completed with the native-opponent Q reference. The loss, sampler, buffers, schedules, target inference, diagnostics, evaluation and checkpoint paths are the actual multiround implementation.

| Learner | Reuse | Phase | Fresh records | Seconds including setup | Fresh records/second |
|---|---:|---|---:|---:|---:|
| neural | 1 | pre | 50000 | 27.933 | 1790 |
| neural | 1 | post | 50000 | 24.959 | 2003 |
| neural | 4 | pre | 50000 | 32.773 | 1526 |
| neural | 4 | post | 50000 | 30.476 | 1641 |
| tabular | 1 | pre | 50000 | 26.164 | 1911 |
| tabular | 1 | post | 50000 | 24.076 | 2077 |
| tabular | 4 | pre | 50000 | 32.087 | 1558 |
| tabular | 4 | post | 50000 | 29.831 | 1676 |

Every case has 50,000 records and at least two hard target refreshes. Post profiles exercised supervised updates and learned-average actions. They open at zero and finish hand-off at 1,000 only for timing; they cannot satisfy formal gates. All other production-shaped counters and replay paths remain in force.

| Learner | Reuse | Phase-weighted 15M forecast, hours | Preregistered per-seed ceiling, hours | Complete pipeline |
|---|---:|---:|---:|---|
| tabular | 1 | 2.064 | 4.25 | True |
| tabular | 4 | 2.549 | 5.25 | True |
| neural | 1 | 2.163 | 4.50 | True |
| neural | 4 | 2.603 | 5.25 | True |

The profiler's sequential process wall time was 229.264 seconds. It ran concurrently with the RPS profiles/first screen; these are not isolated timings.

Each forecast weights the pre/post measurement by 5M and 10M records. Ceilings exceed twice the forecast, rounded up to 15-minute units. Setup/evaluation/checkpoint scaling is conservative, but buffers were partially filled and this is not a full-capacity IO benchmark. Host-memory figures in manifests are measured; logical allocation sizes are not resident usage. These CPU costs do not price CUDA, 9×9, or Retrace.

C1 and the paired tabular-control ceiling are now entered for a seed-0 R=1 delayed first screen. C4 is a costed reserve only: a full R=4 comparison is conditional on the declared one-step decision, not automatically selected. Remaining seeds and all other A2 gates remain mandatory before any pilot. The complete full-study H_P/H_D/H_N ledger remains pending large-board profiling; no 9×9 allocation is funded here.
