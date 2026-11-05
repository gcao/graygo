# Persistent learning and champion releases

The controller in tools/champion_loop.py retains Q, target and average networks,
optimizers, replay, reservoir, episode actors, private RNGs and cumulative record
clocks. Rejecting or promoting a release never rolls back this learner or clears
buffers. An imported old champion is an evaluation reference, not a warm start.

## Learning and evaluation budgets

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
historical/baseline regression checks and atomic publication. A rejection-only
futility check stops a clearly unpromising first look before the larger look.
Futility and screening never authorize promotion.

## Q and average-policy diagnostics

tools/champion_screen.py evaluates frozen greedy Q and initialized average
policies separately against random and tactical opponents. Defaults are 20 games
per matchup, paired across colors, with a 30-second total screen ceiling. Reports
include wins/draws/losses, utility, game lengths/endings and empty-board pass
probability. An uninitialized average policy is reported explicitly and skipped.
Screening does not mutate learner parameters, buffers or RNG state.

An initialized average must have positive sample mean utility against both
baselines to proceed to an expensive champion gate. This is a cheap rejection
filter with a small fixed sample, not a confidence or deployment-readiness claim.
Only an average policy can be released; Q diagnostics cannot promote Q.

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
  unchanged. Only initialized averages clearing screening and the independent
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
screens before average initialization, fixed schedule validation, native-process
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
