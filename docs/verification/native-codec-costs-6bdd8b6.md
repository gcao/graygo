# Native compact-decoder costs

Source commit `6bdd8b68b831da664bb3dee7f19e3c6d25df1b7b`, SHA `ef0ffeea609bef788791f4bb92d57da4d18007f20a323b90bef8e2bec6ddfec6`. The immutable library/build identity and original snapshot manifests are preserved in the raw report. Identical 20,000-record source data and matched sampling streams compare feature storage, strict Python rule decoding and fully validated native decoding.

| Decoder/storage | Strict restore seconds | Replay decode seconds | Reservoir decode seconds | Paired allocated arrays, MB |
|---|---:|---:|---:|---:|
| features | 0.0848 | 0.0152 | 0.0044 | 603.5 |
| reference | 6.4483 | 3.9938 | 1.9894 | 203.0 |
| native | 0.1239 | 0.0452 | 0.0201 | 203.0 |

Sampling covers 20 batches of 512 starts; replay regenerates both source and successor observations. Every decoded field and frozen target from the final matched samples agrees across all three paths. Native metadata was written only to a newly owned copy; original snapshots were not modified. All packet/state invariants remain validated.

Native decoding removes the measured Python type/feature-generation bottleneck while preserving the 203 MB logical host-array representation. It is still a partial-fill microbenchmark, not total RSS, full-capacity IO or complete learner cost. Timing was concurrent with the original first screens and is not an isolated machine benchmark. The 600k-record/500k-capacity native IO benchmark has been launched separately; no full-capacity result is claimed here.

Additional native boundary fixtures pass for even boards, the 256-point action-width transition and maximum supported 32-bit round count. The complete suite has 142 passing tests; the extended native fixtures are recorded beside the report. Complete CUDA training/resume, large-board rollout throughput and H_P/H_D/H_N remain pending.
