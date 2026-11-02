# Locked board execution and persistent resource clock

## Mechanism and remaining authority

`BoardExecutor` drives the actual batched board runtime through `BoardMilestoneJournal`. It owns initialization, strict restore, paired chunking, checkpoint cadence, opening stops, disk checks, cumulative process time and finalization while holding one exclusive nonblocking OS lock. The lock stays held through initialization, collection, pending decisions and the final clock write. A competing invocation cannot mutate this allocation.

This component supplies execution mechanics. It does **not** verify project A2, fund a formal scope, grade the whole development cohort or authorize screening/confirmation. The outer project controller must verify those prerequisites and restrict the requested operational endpoint. The public `run-board-diagnostic` command only creates diagnostic declarations, at most 65,536 fresh records and one cumulative hour. It accepts no formal-scope flag. Raw library mechanisms are not approval receipts. The full entry/cohort controller and applicable A2/profile evidence remain required work.

## Exact records, opening and retention

The loop clips each chunk to the earliest of the requested operational endpoint, optimizer boundary, next permanent archive, opening/hand-off boundary and periodic heavy checkpoint. It calls the journal's boundary and label checks before collecting. A missing fixed-opening decision returns `opening_pending` with an unchanged closed reservoir; a sealed no-go returns `opening_no_go`. The loop never changes the exploration clock, copies an older winner, clears labels or opens automatically from a record count. Confirmation's fixed label schedule remains the journal's responsibility.

Milestone capture and periodic saves use the real current runtime, including optimizers, target, all episode-frozen policies, private RNGs and full buffers. Interrupted/time-capped work keeps the two-generation recovery discipline. Only the true run horizon or a sealed opening no-go finalizes to one heavy generation; all public weight archives stay permanent. `until_records` is a stop within the original horizon, not a replacement horizon or permission to advance a cohort.

## Frozen contract and cumulative time

One atomic checksummed `execution.json` binds source, runtime config, numerical hardware/library/settings, native decoder identity, starting distribution, journal plan, checkpoint cadence, resource ceiling and disk reserve. The state binds the initialized run and latest durable bundle by checksum. Resume requires the original contract; changing source, hardware/runtime, starts, cadence or ceiling cannot refresh this allocation. Missing clock or initialized manifest fails instead of inventing a new run.

Measured CLI setup is charged on every invocation, followed by initialization/restore, collection/learning, in-context validation and IO. Interpreter startup before the CLI timer, standalone reporting/audit and work by separately launched evaluators are separate overhead; the outer project ledger must account for them. When a session ends cleanly, offline time is uncharged. A heartbeat preserves observed time and an in-flight wall timestamp. An unclean restart conservatively charges the unobserved interval, including offline time, up to the remaining ceiling. Backward wall time exhausts the remaining allocation rather than adding it. Atomic initialization, one chunk and safe IO can overrun a check boundary; actual observed time remains visible.

The execution lock covers all clock reads/writes. Clock replacement fsyncs both its payload and parent directory. A new checkpoint commits before its execution-clock pointer. If a crash lands between those commits, strict journal restore can adopt the newer verified bundle. A clock-recorded bundle may not silently disappear or change; fallback to an older checkpoint does not erase the durable clock. Initialization interrupted before its first bundle can reproduce and seal record zero under its original manifest.

## Disk and operational failure

Before each collection chunk, free space must cover the frozen headroom plus reserve. Headroom is at least the declared amount and a conservative full-buffer/model/snapshot/optimizer estimate, even when the live buffers are nearly empty. Each actual save also checks its own projected peak before writing. Formal journals retain the twenty-GiB minimum reserve; the public diagnostic CLI offers no reserve override.

A low-disk stop reports the last verified durable bundle and any unsaved suffix. It does not lower the reserve, delete permanent archives, truncate a bundle or finalize a partial run. If the clock itself cannot be written, the prior in-flight state remains for conservative recovery. An unexpected numerical/operational exception leaves the previous durable learner state intact; work already spent remains charged and the comparison is incomplete.

## CUDA restore correction

The new live-state continuation check compares tensor device placement as well as values/dtypes, optimizer/target state, collector state and all RNGs. It exposed that mapping an entire runtime bundle to CUDA also moved Adam's CPU step counters onto CUDA. Restore now reads storage on CPU and lets model/optimizer loaders place their respective tensors. The full live CPU/CUDA state then matches uninterrupted execution exactly.

Historical saved-state audits loaded both sides onto CPU and therefore proved stored values/RNG contents, without independently checking live optimizer-step placement. Their source-bound artifacts remain preserved. The corrected source's stronger test covers the live layout explicitly. CPU numerical metadata also avoids querying cuDNN in a way that creates an unintended CUDA context.

## Commands and diagnostic scope

```sh
# Start a fixed 16,384-record diagnostic; stop at its unchanged 4,096 prefix.
.venv/bin/graygo run-board-diagnostic --run-dir runs/board-execution-example --records 16384 --until-records 4096 --time-cap-hours .08333333333333333 --device cuda --q-reuse 4
# Resume the same ceiling/config; collection stops at the diagnostic opening.
.venv/bin/graygo run-board-diagnostic --run-dir runs/board-execution-example --records 16384 --time-cap-hours .08333333333333333 --device cuda --q-reuse 4 --resume
# Explicit diagnostic control only; cannot issue a development-pilot decision.
.venv/bin/graygo run-board-diagnostic --run-dir runs/board-execution-example --records 16384 --time-cap-hours .08333333333333333 --device cuda --q-reuse 4 --resume --opening-decision pass
```

The default diagnostic opening/hand-off are 8,192/12,288, deliberately reduced from the formal 5M/7M schedule. Native decoding is explicit via `--native-library`; that identical path/checksum/ABI is required on resume. To run the bounded real-CLI/native integration and independent saved-artifact audit:

```sh
.venv/bin/python tools/check_board_execution.py runs/board-execution-check --native-library artifacts/native/acfca79ab9de20d01dbb/libgraygo_rule_decode.so
.venv/bin/python tools/check_board_execution.py runs/board-execution-check --audit-only
```

The driver requires clean committed package source, two trajectories of 16,384 records (32,768 aggregate), and a 300-second persistent ceiling per trajectory. One runs uninterrupted; the other uses three actual CLI processes with a prefix stop and pending opening before completion. Independent checks cover source/native/bundle/clock/control bindings, all permanent archives, label history, optimizer counts and exact complete stored runtime/buffers. No A2, formal pilot, campaign pricing or trained-strength conclusion follows from this integration check. Actual collected evidence is appended after its terminal audit.


## Test result

The final combined suite passes **245 tests in 107.462 seconds**. Eighteen execution-specific tests include exact live CPU/CUDA continuation, private RNGs and complete arrays; opening pass/no-go retention; lifetime locking; bounded CLI rejection; cumulative setup/runtime ceilings; clean versus unclean/offline/reversed-wall handling; missing/corrupt provenance; bundle-before-clock crash recovery; full-capacity write headroom; low-disk stops; and safe durable state after unexpected exceptions.
