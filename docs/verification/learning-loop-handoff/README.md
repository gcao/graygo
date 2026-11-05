# Learning loop handoff verification

Eighty tests passed, covering the unchanged learning/evaluation checks plus
complete-state handoff, single-use budget transfer, publication continuity,
unchanged parent checkpoints, source compatibility and bounded loop entry.

The real stopped parent is the November 5 comparison learner at 397,440 records.
Its numerical source is compatible with this controller upgrade. The child keeps
the 5M/7M averaging schedule and enables experimental independent gatekeeping and
atomic publication to the existing arena manifest. No Q-only warmup checkpoint
is eligible for promotion.

The parent checkpoint, clock and source are retained. The child plan records its
source/checkpoint fingerprints and subtracts already-used execution time plus a
conservative setup charge from the original one-hour allocation.

The first live collection exposed a restored-snapshot identity collision. Its
397,440-record checkpoint was preserved and compared bit-for-bit (excluding
timings) with the stopped comparison checkpoint. No training state was lost.

The runtime now reuses the restored frozen object for the current policy version
when a new episode joins active episodes. A regression resumes with one active
and one empty environment before the first optimizer update, then checks exact
agreement with uninterrupted training. All 81 relevant tests pass.

Source compatibility admits only this exact cache-canonicalization insertion in
addition to the prior controller/source-admission changes. A replacement stage
deducts the failed stage's observed time and another setup reserve from the same
original allocation.
