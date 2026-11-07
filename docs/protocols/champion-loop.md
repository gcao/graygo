# Persistent learning and champion releases

## Promotion policy v3 (owner approved November 6)

Promotion now requires a statistically convincing head-to-head win against the
current champion, no material historical regression, and the existing absolute
strength floors. Pool improvement remains a reported diagnostic and cannot veto
promotion or terminate a gate for futility. The remaining thresholds, confidence
families, alpha allocation, fixed looks and independent private actors are
unchanged. The aggregate endpoint remains in the confidence family so existing
bounds reproduce exactly and future bounds are not weakened.

This corrects the saturated-opponent problem: an incumbent scoring 97.1% against
an older model has only 2.9 percentage points of possible improvement left.
Requiring statistically positive additional gains there can block a candidate
that convincingly beats the incumbent directly.

New gates use a head-to-head-history-v3 directory. Completed original gates and
their evidence remain immutable. tools/adopt_head_to_head_policy.py reassesses
complete sealed evidence, reproduces every original confidence bound and tests
the three retained safeguards before an offline publication upgrade. It verifies
that the evaluated incumbent is still current, preserves the learner checkpoint,
optimizers, replay, RNG and historical decisions, retains the prior champion in
history, and resets only the adaptive evaluation reference/cooldown. A pending
gate whose incumbent changed is retired without reusing its confidence slot.

The owner requested this correction after the 22,544,384-record candidate scored
74.3% head-to-head against the incumbent and passed the other safeguards, while
its additional gain against the old model was too small to prove statistically.

## Historical promotion policy v2

The pool-improvement rule now requires its lower confidence bound to be strictly
positive. The previous minimum of +0.03 utility is retained as the explicit legacy
policy minimum-pool-gain/v1 for reproducing historical decisions. Head-to-head
confidence above 50%, historical regression bounds, absolute floors, planned
sample looks, confidence allocation and private actor sampling are unchanged.

New evaluations use a positive-pool-gain-v2 subdirectory so historical plans and
databases retain their original meaning. The tested 7,340,032-record snapshot is
reassessed from its complete 2,000-game-per-pairing evidence. The original failed
decision remains intact; a separate owner-approved assessment records the policy
change and successful result.

tools/adopt_positive_gain_policy.py verifies evidence hashes, reproduces all
original statistics, verifies immutable model identities and requires the
evaluated incumbent to still be current. Offline application archives mutable
controller plans/contracts, updates the selected champion pointer atomically,
retains the previous champion as history and rebases any unfinished candidate on
the new incumbent. The current learner, optimizer, replay, counters and RNG state
are untouched. New games resolve the new model; existing games keep their pinned
snapshot.

## Continuous training (current operation)

The owner removed the training budget limit on November 5. The managed
continuous-training runner supersedes the earlier one-hour allocation. It has no
wall-clock or total-record stopping limit. It carries the complete saved learner
forward through successive 20M-record archive segments; 20M, 40M and later
boundaries trigger automatic continuation.

The user service graygo-learning-loop.service runs a pinned source checkout,
continues across SSH logout/reboots, and restarts from durable checkpoints on
unexpected process failure. SIGTERM stops cleanly. Storage/numerical integrity
guards remain active. A storage-limited result stops for intervention instead of
restarting repeatedly.

Q, target, optimizer, replay, reservoir, active episode actors and RNG state are
preserved. The 5M opening, 5M–7M handoff, 40M epsilon clock and fresh-record target
refresh never restart at a segment boundary. Only the execution clock and archive
controller gain unlimited operation. Historical elapsed and evaluation time also
carry forward.

Unlimited campaigns use the adaptive evaluation schedule below. Evaluation time
remains recorded, but cumulative evaluation debt no longer blocks a full gate.
A started gate retains its frozen candidate and evidence through finite one-hour
sessions until a decision is reached. Bounded diagnostics retain their original
hard time ceilings and evaluation allowance. Gate thresholds are unchanged.
New segments retain their distinct evaluation seeds and summable alpha allocation
.05/((i+1)*(i+2)); the continuous campaign total is at most .05. Existing formal
scientific/cohort grants remain separate.

The hourly status timer reads live JSON heartbeat counters and continues logging
to runs/training-watch.log. Its RUNNING/STOPPED status follows the continuous
controller, and each entry distinguishes live records from durable checkpoint
records.

Operator commands on ai:

~~~sh
systemctl --user status graygo-learning-loop.service
systemctl --user stop graygo-learning-loop.service
systemctl --user start graygo-learning-loop.service
tail -F ~/proj/graygo/runs/training-watch.log
~~~

Service configuration is in deploy/systemd/graygo-learning-loop.service.
runs/continuous-training.env selects the pinned entrypoint, campaign directory
and PYTHONPATH; it contains paths only.

The controller in tools/champion_loop.py retains Q, target and average networks,
optimizers, replay, reservoir, episode actors, private RNGs and cumulative record
clocks. Rejecting or promoting a release never rolls back this learner or clears
buffers. An imported old champion is an evaluation reference, not a warm start.

## Bounded diagnostic evaluation budgets

The loop reserves at most 10% of its cumulative wall-clock ceiling for screening
and promotion evaluation together. This charge survives checkpoint resume and
unclean evaluation recovery. Each screen/gate also has its own time ceiling.
Evaluation finishes complete color pairs, so one in-flight pair and safe IO can
overrun a soft allocation. Long experiments require an outer hard watchdog.

The public command rejects an incompletely evaluated candidate without promotion
and continues learning. It does not keep retrying the same candidate until the
entire training budget has gone. The API can still request resumable pending
evaluations through continue_after_evaluation_budget=False.

The full gate retains independent actors, fixed looks, confidence allocation,
historical-model regression checks and atomic publication. A rejection-only
futility check stops a clearly unpromising first look before the larger look.
Futility and screening never authorize promotion.

## Adaptive gate scheduling (owner approved November 6)

Unlimited campaigns adapt the amount of training between evaluation attempts.
They no longer require the cumulative ten-percent evaluation account to catch up.
The schedule counts learning/execution seconds with all recorded evaluation time
excluded, so a long gate does not consume the subsequent training interval.

- A complete screen with at least 70% win score against the champion and stable
  historical results can trigger a full gate at the next checkpoint. The interval
  shortens toward a minimum of 30 minutes. Historical stability requires at least
  45% score and no drop greater than 15 percentage points against the same saved
  model since its previous screen. These are screening heuristics only.
- Modest improvement gets 30–60 minutes of training before another check. Two
  flat/losing screens lengthen the interval, doubling up to four hours. Progress
  checks remain at most one hour of training apart. An eligible modestly winning
  candidate cannot have its full gate deferred beyond four hours of training.
- Completed unsuccessful gates lengthen the training interval. The same strong
  screen score does not repeatedly override that wait; a new meaningful gain can.
  After promotion the new champion becomes the reference and the first check is
  scheduled after 30 minutes of training.
- Incomplete gates always resume the same frozen candidate and saved color pairs.
  The heuristic cannot promote a model, change gate thresholds, add statistical
  looks, or reuse an already spent confidence slot. Skipped checkpoint candidates
  remain skipped; their evaluations are not retried until a favorable sample.

Schedule state and its implementation identity are persisted and carried through
restarts and archive segments. tools/upgrade_gate_scheduler.py --adaptive-schedule
admits only the scheduling tools, verifies unchanged numerical source, archives
operational metadata and preserves checkpoint payloads, models and started gates.
The hourly log exposes gate_schedule, interval, signal, screen_due_in and
gate_timer_in. A strong screen may trigger a gate before that ordinary timer.

## Model-only evaluation (owner approved November 6)

The active learning/release loop no longer schedules random or tactical bots.
NFSP collection remains self-play; exploration, replay, reservoir, optimizers,
private action RNGs and the averaging schedule are unchanged.

tools/champion_screen.py screens the frozen average policy against the current
champion and up to three frozen historical models, with 20 color-paired games
per opponent and a 120-second total screen ceiling. Q reports its opening pass
probability without playing separate diagnostic matches. Before average-policy
initialization no strength matchups are run. A positive sample mean against the
champion is a rejection-only filter; historical matchups are descriptive here.
A screen cannot promote a model. Unlimited runs use the adaptive schedule above;
its screen and gate costs remain in the evaluation ledger for observability.
Full gates retain their confidence bounds,
head-to-head requirement, historical regression checks and absolute floors. Their pool now contains historical models only, with the frozen
bootstrap model as the initial reference if no previous champion exists.

Reports use a separate champion-first-v2 namespace; original baseline screens
and promotion evidence remain intact. A stopped campaign can adopt the schedule
with tools/upgrade_gate_scheduler.py --evaluation-schedule. The upgrade verifies
unchanged numerical source and checkpoints, archives controller metadata, and
updates only the pinned tools and operational contracts. An unfinished old gate
is retired without promotion; its confidence slot and evidence are not reused.

The hourly observer leads with the champion and learned-opponent results, names
the opponent actually tested, and reports benchmark_records, benchmark_age,
benchmark_measured_at and lag_records. It carries the latest measurement across
archive segments and distinguishes screening from an in-progress/full gate.
A repeated historical result is not a new measurement of the current learner.

## Preserve the learning schedule

Normal runs keep the designed reservoir opening at 5M records and the 5M–7M
episode handoff. Epsilon retains its fixed 40M clock. A short run should report Q
progress while the average network remains uninitialized, rather than force an
average network to imitate an immature Q policy.

Three bounded public modes are available:

* --diagnostic: at most 65,536 records and one cumulative hour. A changed opening
  requires the additional explicit --early-opening-diagnostic flag.
* --practice: a nonformal Q-learning prefix of at most 5M records and one
  cumulative hour, with the primary opening/handoff unchanged. It stops at or
  before opening, with no reservoir labels or average-policy promotion. This
  permits useful bounded learning without calling a short diagnostic a completed
  NFSP campaign. It does not issue A2, cohort, formal training or promotion grants.
* --loop: a bounded experimental learning/release loop through at most 20M records
  and one cumulative hour per explicitly allocated stage. It retains the fixed
  5M opening and 5M–7M handoff and can therefore train the average policy and
  promote it. Like the fixed-opening confirmation recipe, opening is not filtered
  by the per-seed development window. This mode makes no A2/cohort/confirmation
  claim. Existing formal development-window and confirmation controls remain
  unchanged. Only initialized averages clearing model-only screening and the independent
  promotion gate can update the selected arena manifest.

Formal development/confirmation configuration, readiness decisions and calibrated
promotion authority remain separate. Neither public mode bypasses those gates.

Example standalone prefix (not launched merely by reading this document):

~~~sh
PYTHONPATH=src:. .venv/bin/python -m tools.champion_loop \
  --practice --run-dir runs/q-prefix --records 5000000 \
  --block-records 262144 --time-cap-seconds 3600 --device cuda \
  --incumbent-release /home/gcao/proj/graygo-arena/.runtime/models/champion.json
~~~

Publication defaults to the run directory. The arena pointer changes only when
explicitly selected and a candidate clears its full release requirements.

## Matched wall-clock comparison

tools/training_comparison.py separates preparation from launch. Preparation
freezes the actual legacy libraries/native extensions and an evaluation-only
champion checkpoint, hashes the implementation and writes a plan. It does not
start training, modify graygo-old, or invoke its run.py/notifications.

A one-hour total plan gives each fresh learner 1,620 seconds (27 minutes) and
reserves 360 seconds for common evaluation. The new arm uses the primary
averaging schedule and Q reuse R=4. Its internal evaluation allowance is zero
because evaluation is shared at the end. The legacy arm uses its policy, value
and auxiliary losses, 900 search visits, two games per collection chunk, and
persistent optimizer/replay. Small chunks make completed work durable within a
short budget. These choices are recorded rather than presented as an identical
reproduction of the old long campaign.

Both arm ceilings include process initialization, collection, fitting, exports
and checkpoint IO. The comparison uses process-group watchdogs around each arm
and the overall experiment, including native search that ignores a soft stop.
An exclusive launch receipt prevents a repeated command from buying another
allocation. A killed native chunk can lose its in-flight samples; the last
completed checkpoint remains available. The legacy arm stores private replay in
resume.pt and separately publishes strict-loadable weights in latest.pt.

Common evaluation uses the current authoritative engine, private action draws,
raw neural policies and fixed color pairs. It compares new Q, an average policy
if initialized, the freshly trained legacy policy and the existing champion
reference. It also includes a direct new-Q versus fresh-legacy matchup. No arena
publication occurs. Partial evaluations, actual work/update counts and failed
arms are retained.

The old trainer keeps its native training rules and larger architecture. Thus
this is a practical stack comparison under matched wall-clock ceilings, not a
controlled experiment isolating the learner algorithm. One seed, short training,
different game lengths and shared-machine load further limit conclusions.

~~~sh
PYTHONPATH=src:. .venv/bin/python -m tools.training_comparison --prepare \
  --run-dir runs/training-comparison-1h \
  --old-repo /home/gcao/proj/graygo-old \
  --reference-checkpoint /home/gcao/proj/graygo-arena/.runtime/models/graygo-old-champion-0081.pt \
  --total-seconds 3600 --device cuda

# Explicitly launch the prepared allocation:
PYTHONPATH=src:. .venv/bin/python -m tools.training_comparison --run \
  --run-dir runs/training-comparison-1h
~~~

## Retained first diagnostic

The November 3 one-hour diagnostic retained a 16,384-record learner with 100 Q
updates and 26 supervised updates. Collection/learning took approximately 59
seconds; evaluation consumed most of the run. The candidate was not promoted.
Its zero-opening/1,000-record handoff was a diagnostic override, not the primary
schedule. That run is immutable; the revised controller does not reset its cap,
reinterpret its evidence or overwrite its release.

Tests cover learner preservation on rejection/promotion, resume, evaluation
quota recovery, continued learning after evaluation exhaustion, independent Q
diagnostics before average initialization, fixed schedule validation, native-process
deadline enforcement, source snapshots and existing publication guarantees.

## Switching a running allocation to the learning loop

Stop the comparison learner gracefully before switching, so its latest full
checkpoint and execution clock are closed. tools/learning_loop.py prepares a
single-use parent handoff and launches the enabled loop with the unspent budget.
The parent checkpoint, journal and source remain immutable. The child manifest
records its parent checkpoint seal, source compatibility proof, inherited record
clock, previously used seconds and remaining allocation.

The handoff restores the complete model/target/optimizer state, replay, reservoir,
active frozen episode actors, collector and private RNGs. Exploration and target
refresh clocks continue from their cumulative counters. A source upgrade is
admitted only when numerical source files match and the runtime's state-loading
body is identical; controller/scope changes and the explicit source-admission
guard are recorded separately. Ordinary resume still requires an exact source
match. Cross-version handoffs with an initialized average are rejected because
their old deployed-policy archive identities require separate reconciliation.

The parent execution lock stays held while the child runs. A single-use parent
reservation and launch receipt prevent multiple children from spending the same
remaining allocation. A process-group watchdog stops the stage at its ceiling.
The arena manifest is explicitly selected; successful candidates affect new
games through the existing atomic publisher. No promotion happens during Q-only
warmup before average-policy initialization.

Restored episode actors and newly started episodes share one canonical frozen
snapshot per policy version. The source-compatibility verifier admits the exact
cache-recovery insertion for older checkpoints; it still rejects other learner
or model changes. Failed handoff attempts are charged before another stage.

## Hourly status log

The independent user timer graygo-training-watch.timer runs
tools/log_training_status.py at the top of every hour. It reads the latest loop
receipt, execution/league metadata and screening summaries, then appends to
runs/training-watch.log. Entries explicitly distinguish running, stopped and
missing-supervisor states. Stopped/budget-finished runs continue to produce an
hourly status entry.

The logger uses standard-library JSON readers and does not import a training
runtime. Its lifetime is independent of the bounded run-local watcher. The
installed user timer is persistent across SSH logout on ai, where user lingering
is already enabled. The source unit files are in deploy/systemd/.

Tail the stable append-only log:

~~~sh
ssh ai 'tail -F ~/proj/graygo/runs/training-watch.log'
~~~

Inspect the schedule or request an immediate status entry:

~~~sh
systemctl --user list-timers graygo-training-watch.timer
systemctl --user start graygo-training-watch.service
~~~


## Historical gate-completion fix

The guarantees about frozen candidates and retained evidence still apply. The
current unlimited scheduler uses adaptive timing above; the cumulative debt
behavior described below records the earlier scheduler.

The initial sustained scheduler incorrectly applied its earned time allowance as
a hard cancellation inside every full gate. It then rejected the incomplete
candidate and started a new one at the next checkpoint. This could repeatedly
collect only tens of head-to-head games instead of finishing the registered
500/2,000-game looks. It was an operational timeout, not a completed statistical
rejection.

Unlimited runs now finish a started gate, reusing its frozen candidate, incumbent,
pool, confidence family, seeds and committed color pairs across sessions and
restarts. Completing a gate can temporarily exceed the ten-percent target. The
resulting debt defers future evaluations while learning continues. Screening,
head-to-head, pool-gain, regression and absolute-floor thresholds are unchanged.

A tools-only upgrade can be applied offline with tools/upgrade_gate_scheduler.py.
It requires stopped execution and publisher locks, admits only a changed
champion_loop.py implementation, and archives the original plans and contracts.
Checkpoint/optimizer payloads and evaluation databases remain untouched. A pinned
source checkout and operational fingerprints are amended explicitly so the
existing campaign can resume.

Hourly logs include full gate counts such as gate_head=72/500 and any completed
gate's failed criteria, separately from the small screening results.
