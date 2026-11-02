# Complete delayed-task profile launch

Source commit: `94a38602d251960755ce515aa63dff17fc2e74ac`. Source SHA-256: `1b0fd1521a04f8f0ca63ad5b331b89802cacdd337382b25b86689a458d8e934d`. The immutable pinned worktree is `.worktrees/nfsp-tasks-94a3860`.

The plan beside this document runs eight CPU cases: neural/tabular × one-step R=1/R=4 × pre/post opening, 50,000 fresh records each. They use the production 512-start update cadence, 4,096-record warm-up, 12,800-record target refresh, 500k buffer capacities, opaque 50/100/200-round states, 40 equivalent intermediate choices and complete balanced supplemental coverage. Both current/frozen next-state target inference and actual diagnostic overlap computation are included. Exact evaluation and checksummed full buffer/active-episode checkpoints occur at the midpoint and endpoint.

Pre-opening profiles retain the primary closed reservoir. Post-opening profiles deliberately open at zero and finish the hand-off at 1,000 records to exercise supervised/average-policy work. These are timing recipes, not alternate scientific candidates, and cannot pass formal gates. Each profile has a 15-minute operational ceiling; the eight-case profiling ceiling is two CPU process-hours. A full delayed-family allocation is not launched by this plan.

The profiler emits per-learner/reuse budgets only after both phases complete with actual Q updates, at least two target refreshes, supervised updates and learned-average acting. Its phase-weighted 15M forecast uses a twofold margin and records partial-buffer and scaled evaluation/startup limitations. These CPU costs do not price 9×9 or Retrace. C1/C4 remain provisional until the completed reports are inspected and entered into the ledger.

Launch artifacts are `/home/gcao/proj/graygo/runs/nfsp-task-profiles-94a3860` on `ai`; original PID 2208745. It launched concurrently with the existing asymmetric matrix screen (PID 2193858), so do not describe the timing as isolated. The live OS process, not this historical PID or a state file, determines whether a job still runs. Inspect the original handle before any resume; an observation timeout is not permission to restart.
