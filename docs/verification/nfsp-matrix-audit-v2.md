# Matrix audit v2: publication history and evidence bindings

The independent matrix auditor now reconstructs the actual last published tabular policy and emits input-bound receipts suitable for later cohort assessment. Its changes affect verification tooling only. The learner package remains `9dbe715f543b2067a078693c943254010602616defaf5e3cecf81fe23ab5e10f`; the live corrective trial still uses its separate frozen `ca45a53` package. Historical audit reports and frozen script copies remain unchanged.

## Reproduced defect

The original auditor compared the published table directly with the latest reservoir. A run can finish between supervised update boundaries. Eligible labels arriving after the last publication change the reservoir without changing the deployed table, so that comparison can reject a correct checkpoint.

An unchanged-trainer pure82 development fixture reproduces this at **254 aggregate records**, batch eight, seed two and executed-action labels. Its last publication is at **248**. The published per-role counts are **9 and 5**, while the latest reservoir has fifteen labels, including one newer black label. The correct black conditional mean includes nine labels; the original auditor instead demanded the ten-label mean and rejected it, with a maximum probability difference of 0.1. This is a verifier defect, not evidence that the learner published an incorrect table. Existing aligned-endpoint tests did not expose it. The formal 15M matrix endpoint is also between 128-record boundaries, so publication history cannot be assumed away.

Version two replays exact Algorithm-R slot changes and reverses changes after the last supervised publication. For executed-label controls, the permanent matrix episode log retains every actual action, allowing exact reconstruction even when an old label was replaced. For greedy labels, unchanged targets are exact; discarded replaced greedy targets remain unknown. Their original role/count is known, so the verifier checks necessary simplex bounds and normalization, lists exact roles and reports whole-table verification false when any target remains unresolved. It does not invent a missing greedy distribution or silently replace it with current Q.

## Stronger receipt and raw checks

Every scheduled public model archive is checksum-verified and bound in the receipt, along with the original run manifest, result, completed resume bundle, episode log and auditor source. The auditor recomputes all scheduled original-game average/greedy gaps and readiness, validates their aggregate/Q/update/refresh/supervised clocks and preserves all fixed final-five criteria. Final public Q/average tensors must equal the final learner tensors.

Complete permanent episode actions, mode/ramp choices, independent RNG streams, eligible positions, Algorithm-R replacements and label versions are verified. Retained replay ordinals must cover the complete latest interval without duplicates. Packed role/legal/greedy arrays must be canonical; executed-label targets must exactly equal actual logged actions. The verifier reconstructs both roles' private inverse-CDF actions and behavior probabilities through the next update batch after each archived model, and checks retained greedy targets covered by those frozen parameters.

Read-only entry holds a shared lock on the existing trainer lock, uses static checkpoint verification and creates no training directories. Pinned dependencies are required. CPU model reconstruction preserves the caller's CPU/CUDA RNG and thread settings, including an ambient CUDA default device. The CLI protects original training files from its output path and has an explicit watchdog, at most four elapsed hours. That watchdog is not an aggregate CPU-stage allocation; imports, tests, development and other overhead remain separately scoped.

```sh
PYTHONPATH=ORIGINAL_FROZEN_SOURCE/src .venv/bin/python tools/audit_nfsp_matrix.py ORIGINAL_RUN \
  --time-cap-seconds 600 --output artifacts/matrix-audit-v2.json
```

Reduced fixtures require `--allow-diagnostic`. A verified diagnostic or single task/seed does not complete A2. Full historical model gradients and every historical frozen greedy set are not reconstructed; the receipt states its exact publication and history limits.

## Validation

The combined suite passes **332 tests in 264.905 seconds**, including twelve new matrix receipt tests. The tests reproduce the valid unpublished executed-label tail; exercise a genuinely unresolved replaced greedy label; verify archive/checkpoint/result bindings and real CLI behavior; and reject resealed action-label, parameter, ordinal, padding, scheduled-clock and dependency corruption. They also verify writer exclusion and CPU/CUDA RNG/thread preservation. Existing matrix, delayed, bounded, native decoder, paired evaluation and board continuation tests remain green.

The [full log](nfsp-matrix-audit-v2-tests.txt) and [validation record](nfsp-matrix-audit-v2-validation.json) preserve source fingerprints. A separately declared [read-only historical recheck](../experiments/matrix-audit-v2-historical-recheck-allocation.json) can verify the upgraded receipt against one original full matrix trial; it funds no new learning or automatic cross-source reuse. Actual three-seed A2 evidence, resource decisions and the production entry controller remain required.
