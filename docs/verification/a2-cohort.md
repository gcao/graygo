# Audited NFSP A2 cohort assessment

`tools/a2_cohort.py` combines source-bound finite-case evidence under the design's fixed A2 criteria. It supplies three commands: create an immutable numerical plan, freshly audit one original run into a retained case snapshot, and assess the available cohort. It does not create learners, allocate cases or authorize an A2 exit, extension, pilot or campaign.

## Fixed roster and protocol

The plan names the learner package, exact collector and independent-auditor fingerprints, one-step target and selected multiround Q reuse. It fixes seeds **0/1/2** for asymmetric, RPS, pure bounded Go, mixed bounded Go, delayed and meaningful-branch tasks, retaining neural and matched tabular slots: 36 required evidence slots. Missing slots stay visible. This roster is a verification inventory, not funding for 36 runs or an enlargement of any family/control allocation. No case is started or replaced by these commands.

Matrix adapters keep their declared R=1/128-target protocol; multiround adapters use the explicitly selected R=1 or R=4/512-target recipe. Task-specific differences remain explicit. Primary settings are checked with the pinned trainer's formal recipe, fixed 5M opening, 5M–7M hand-off, 7.5M A2 epsilon clock, 12,800 eligible-Q refresh and full 15M endpoint. Forced-only chains, matching pennies, pure82 controls, reduced diagnostics and changed settings do not substitute for required primary families. Comparator/fixture evidence remains a separate A2 prerequisite.

One plan currently requires one exact learner package across its cases. A historical result from another implementation cannot be silently relabeled, even if it has the same numerical gap. Any cross-source reuse needs its separately recorded compatibility/scope decision. The collector and auditor tools can run from a clean tool checkout with the original learner package on `PYTHONPATH`; adding assessment tooling need not alter the frozen learner itself.

## Fresh case collection and retained inputs

`audit-case` holds a shared nonblocking lock on the existing trainer lock. It binds the original manifest/result/checkpoint, fingerprints all input files and directory inventory, then invokes the plan's exact independent matrix, delayed or bounded auditor. Bounded audits also bind their shared chain helper. All input fingerprints must remain unchanged. Output lies outside original training inputs, and an existing case snapshot is never overwritten. An incomplete snapshot is retained when a later completed snapshot is collected under a new filename.

The snapshot carries independently recomputed scheduled metrics, the complete upstream audit receipt and all input hashes. Source, original family/learner/seed, recipe, endpoint and checkpoint bindings are checked. CPU verification preserves caller RNG/thread state. A completed primary snapshot requires a fresh independent audit; a time-capped or interrupted run records its sealed incomplete state without claiming that its reported policy rows have been independently regraded. An active writer prevents collection rather than being treated as stopped.

Ordinary assessment checks all present original inputs while holding shared locks through the report. `--archived` permits portable numerical reanalysis without the private original model files. That report explicitly has `current_input_snapshots_verified=false`. Both modes retain the snapshot's source/plan fingerprints and record the assessor source. Neither mode independently grants original cohort-registration, training-history or resource authority.

## Gates and marginal-window rule

The complete primary evaluation sequence must be exactly 0.5M through 15M, every 0.5M. The final five are fixed at **13M, 13.5M, 14M, 14.5M and 15M**. A deployed average cannot be initialized before the fixed 5M opening. Uninitialized final averages cannot pass even if a reported gap is small. No earlier winning look, hidden later look or replacement seed can stand in for this window.

Short-game deployed-average gaps must be at most **0.05**. Delayed credit separately requires greedy-Q gap at most **0.05** and supported value error at most **0.10**, with a deployed-average readiness bar **0.10**. Meaningful branching checks both credit and its short-game average. Neural average gaps above 0.10 through 0.25 produce an averaging-lag band only when the other required neural gates pass; failures of mixed tasks cannot be traded for a better chain. Controls are reported separately, including misses requiring interpretation. A missing control, incomplete run or budget limit is not converted into a neural convergence theorem.

The numerical extension check requires a complete original neural cohort, exactly one failing family/seed, a final evaluation passing every relevant criterion, initialized averages, and all earlier misses no more than 20% above their limits. A qualifying proposal names **all three original paired seeds**, their case hashes, at most 2M additional records each and the fixed 17M endpoint. It never selects or funds the extension. An optional predeclared budget hash is a reference, not verification that resources or original-cohort authority exist. Actual extension execution/validation belongs to the project controller.

Neural requirements, matched controls, incomplete cases and extension eligibility are distinct report fields. Even a synthetic all-pass cohort has `a2_complete=false`, `pilot_authorized=false` and `extension_authorized=false`. Actual entry still requires appropriate original-run registration/history, comparator checks, 9x9 smoke, selected-source cost profiles, a filled resource ledger and a recorded scope decision. This component deliberately provides numerical evidence to that controller rather than implementing an implicit launch policy.

## Commands

```sh
.venv/bin/python tools/a2_cohort.py plan --name chosen-one-step \
  --learner-source-sha PACKAGE_SHA --reuse 1 --output artifacts/a2-plan.json

PYTHONPATH=ORIGINAL_FROZEN_LEARNER/src .venv/bin/python tools/a2_cohort.py audit-case \
  --plan artifacts/a2-plan.json --run-dir ORIGINAL_RUN \
  --time-cap-seconds 600 --output artifacts/delayed-tabular-seed0.json

.venv/bin/python tools/a2_cohort.py assess --plan artifacts/a2-plan.json \
  --case artifacts/delayed-tabular-seed0.json --output artifacts/a2-assessment.json

# Portable reanalysis of retained snapshots; does not verify current inputs.
.venv/bin/python tools/a2_cohort.py assess --plan artifacts/a2-plan.json \
  --case artifacts/delayed-tabular-seed0.json --archived
```

Pass `--case` for every available snapshot. With none, the report lists all 36 missing slots. Explicit reduced integration fixtures require `--allow-diagnostic` during case collection and cannot count as primary evidence. The audit watchdog is required and at most four hours; it does not renew aggregate CPU-stage allocations. Case collection never resumes an original learner.

## Verification

Protocol tests use explicitly synthetic cohorts to cover exact thresholds, missing/aliased seeds, separate credit and averaging bands, short mixed-game failures, time caps, uninitialized averages and the exact marginal-window boundary. Real reduced matrix, delayed and bounded runs exercise all three independent auditors, unchanged input snapshots, stale-input rejection, writer exclusion, retained incomplete cases, source-bound CLI operation and archived reanalysis. Those reduced fixtures do not supply formal A2 evidence. The already running corrective neural trial remains on its original source and allocation.

The final suite passes **350 tests in 271.248 seconds**, including seventeen new cohort assessment/snapshot test methods. The [test log](a2-cohort-tests.txt) and [source-bound validation](a2-cohort-validation.json) are retained. A separately [declared one-case integration](../experiments/a2-assessment-integration-allocation.json) uses the already completed corrected control and allocates no new learning.
