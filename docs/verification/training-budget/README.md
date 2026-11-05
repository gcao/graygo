# Training budget repair verification

The retained test log records 76 passing checks over champion control, comparison
budgets, checkpoint execution/journaling, independent evaluation, read-only
evidence and scalar/batched CPU/CUDA parity.

A CPU legacy-arm smoke used 3x3, two rounds, two games, four search visits and a
one-block/eight-channel network. It completed one chunk, collected eight records,
performed four optimizer batches and loaded its public weights through the strict
LegacyPolicy adapter. It is an integration fixture, not strength evidence.

A 60-second CPU supervision smoke exercised hard limits on full-size workers.
It stopped at budget and retained initial checkpoints. A subsequent four-game
common-engine adapter check completed two color pairs. This checks compatibility;
it does not establish training progress or comparative strength.

The one-hour GPU comparison is a separate run with fresh learners, no arena
publication, a fixed 5M/7M averaging schedule, 27 minutes per arm and up to six
minutes for common evaluation. Its actual counters and partial results belong in
its run directory rather than this implementation verification.
