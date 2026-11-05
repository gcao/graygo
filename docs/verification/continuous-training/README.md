# Continuous training verification

All 88 selected tests pass. The added checks cover:
- unlimited execution beyond an hour, with finite JSON clock accounting;
- clean and unclean checkpoint recovery;
- diagnostic scopes retaining their original bounded limits;
- finite evaluation allowances earned during unlimited learning;
- two real archive segments producing the exact learner state of uninterrupted training;
- resuming an interrupted segment without discarding its checkpoint;
- record endpoints beyond 20M/80M and a summable campaign promotion alpha.

The deployed campaign starts from the retained 986,368-record checkpoint. The
new execution/journal controls preserve numerical learner implementation, model
and optimizer state, replay, reservoir, active actors and RNG clocks. The hourly
log reads live heartbeat counters without importing a training runtime.
