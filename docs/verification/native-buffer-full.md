# Full-capacity native-buffer IO

Source commit `6bdd8b68b831da664bb3dee7f19e3c6d25df1b7b`; the library/build identity is in the raw plan. The finite profiler is terminal and completed 600,000 own-side records from legal uniform-reference 9×9 games. Both 500,000-entry buffers are full, with replay wraparound and reservoir replacement. Labels are synthetic all-legal ties; no neural learner, promotion or strength inference is involved.

| Measurement | Result |
|---|---:|
| Logical paired arrays | 203,000,000 bytes |
| Collection | 134.199 seconds |
| Replay sample/decode, 50×512 starts | 0.182 seconds |
| Reservoir sample/decode, 50×512 records | 0.054 seconds |
| Checkpoint write | 3.075 seconds |
| Strict restore validation | 2.790 seconds |
| Checkpoint payload | 24310648 bytes |

Verified decoded sample/target round trips passed. This measures full host-buffer capacity and IO, not total RSS or a complete GPU learner. Actor sampling, target/average optimizer work, inference latency, CUDA memory and evaluation overhead still need their production-shaped measurements before H_P/H_D/H_N can fund a campaign. The original delayed R1 first screen remains separate on its original source.
