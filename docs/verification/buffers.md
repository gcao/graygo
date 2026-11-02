# Compact buffer verification

The replay ring and Algorithm-R greedy-label reservoir passed save/load and checksummed-checkpoint round trips, including RNG continuation. Replay generations prevent stale links after overwrite; sequences preserve player/episode/opponent boundaries. Complete-history evaluation retains episode-component beliefs rather than drawing a new component per state. The combined suite passed 75 tests in 8.970 seconds before this report.

The storage-only profile used 500,000 slots in each buffer, filled 10,000 rows, and decoded 25,600 samples from each. Logical allocated arrays occupy 404,500,000 replay bytes and 199,000,000 reservoir bytes. These are allocation sizes, not measured resident memory. The partial snapshot was 1,471,779 payload bytes and took 0.2744 seconds to write and verify. This does not estimate a full-capacity checkpoint or complete learner cost. The pinned PPO CPU preflight was running concurrently; see the raw report for exact timings/source and that limitation.

The ninth-review schedules are executable helpers: separate logical mode and acting policy, fixed 5M opening and 5M–7M hand-off, independent episode draws, a fixed 40M 9×9 epsilon clock and independent 7.5M A2 clock. Actual NFSP integration and pilot gates remain required.
