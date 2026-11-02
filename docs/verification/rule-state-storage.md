# Complete rule-state storage verification

The full suite passed **134 tests in 41.095 seconds** on `ai`; raw output and source/machine metadata are preserved in `rule-state-storage/`.

## Canonical packet

The 9×9/200-round packet has 104 bytes:

| Field | Bytes |
|---|---:|
| Magic, codec version, rule fingerprint, perspective, history count, pass streak and round counter | 15 |
| Current and two historical boards, two bits per point | 63 |
| Independent black/white restriction bitmaps | 22 |
| Both moves for each historical round, including pass | 4 |

The immutable rule context is explicit in snapshot metadata and bound by the packet fingerprint, including collision mode and round limit. Missing history slots and trailing bit padding have one canonical representation. Player perspective is outside the absolute board/history fields; the decoder preserves complete public identity before generating perspective-relative features. There is no opponent commitment or training-opponent identity in neural inputs.

A state-bound observation constructs features/masks only from its immutable `State` and player. Raw feature arrays cannot silently stand in for a complete state. The codec validates schema/context/padding and delegates resolved-state invariants to the reference engine. Invalid/dead boards, impossible pass/history/counter combinations and wrong rule contexts fail. Sequence padding is explicit and never decoded as a fake game state; true terminal states retain their full packets.

## Evidence

Round trips and regenerated observations match the reference across 3×3, 9×9, 13×13 and 19×19, both collision rules, both perspectives, history/restrictions, terminal mutual passes, geometric/color transforms and counters wider than one byte. Replay wraparound, generation-safe sequence sampling, reservoir replacements and NPZ restore produce the same decoded records/labels as feature storage.

Actual neural Q/target/average updates match bit-for-bit after regenerating features. The bounded mixed-Go trainer also matches complete feature-storage trajectories, weights, metrics, exact evaluations and episode audit bytes through mid-episode checkpoint continuation. The installed CLI exercises compact-buffer save/load. This verifies storage semantics; it is not a convergence or 9×9 strength result.

## Logical memory and performance limits

Default replay uses 275 bytes per entry; the reservoir uses 131. At 500,000 entries each, host arrays total **203,000,000 bytes**, versus 603,500,000 for the existing packed-feature representation, a 66.4% reduction. No duplicate legality/global arrays are stored. These counts omit Python/process/model overhead, transient decoded batches and filesystem caches, and are not resident-memory measurements.

The decoder is currently the Python correctness reference. Sampling regenerates features and strict restore validates states, so lower storage does not imply faster updates. Paired IO/decode profiles, full-capacity restore and the complete production learner must be measured before pricing 9×9. Accelerating the codec is an implementation option; replacing it with an incomplete state or changing learner thresholds is not.
