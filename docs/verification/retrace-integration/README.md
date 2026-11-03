# Conditional Retrace implementation evidence

The shared backend, solved-task trainer, real-board runtime and commands now support the design's explicit lambda-0.95/32-round Retrace candidate. One-step remains the default. [The recorded choice](../../experiments/nfsp-retrace-reuse-choice.json) uses R4 for one conditional comparison, following the full original three-seed R1/R4 misses.

The complete CPU suite passed **620 tests with 22 CUDA skips** at its recorded pre-guard source. A final recipe guard separates formal one-step recognition from formal Retrace recognition, preventing v1 plan relabeling. All **47 affected tests pass** on the final guarded source. [The exact source diff](formal-recipe-guard.patch) and [hash-checked reconstruction scope](guard-scope.json) preserve those distinct test bindings; the 620-test result is not relabeled as a run on a different package.

Four bounded real default pipelines (delayed, meaningful branch, mixed Gray Go and augmented board) match the frozen prior implementation bit-for-bit on the final source: **416 tensors, 126 arrays, private RNGs, episode bytes, counters and evaluations**. Only declared inactive default configuration fields and wall-clock fields are excluded. This finite compatibility check is not a complete source-admission bridge.

Numerical/runtime regressions cover independent rational trace arithmetic, indexing, current-policy/frozen-value separation, one-step reduction, on-policy behavior, true terminal/continuing boundaries, poisoned padding, low-overlap meaningful branches, projection and start-only gradients. Exact neural task/augmented-board checkpoint recovery and a full 9x9/eight-block loss update pass. The first failed prototype staleness check is retained; actor IDs must be decoded to snapshot generations.

The first 8,192-record CLI diagnostic exercised Q work but collected no supervised minibatch. The separate fixed 16,384-record diagnostic exercised **96 Q updates, nine supervised updates, 4,864 average-policy actions and one target refresh**. Its 49,152 loss starts remain distinct from context records. These short-horizon/compact-model diagnostics are not full-learner prices or A2 cases.

An explicit v2 bounded auditor and branching collector independently enumerate complete public pure strategies for native-response grading. The original v1 verifier files and receipts remain unchanged. V1 rejects v2 branching evidence, and the v2 one-step collector rejects Retrace until an independent sequence/optimizer audit exists. The historical cache structure test is pinned to its released source pair rather than implicitly admitting this new package.

[Retention](retention.json) seals original structured logs/reports; [binary inventory](raw-binary-inventory.json) seals raw mutable states, checkpoints and trajectories at permanent diagnostic roots. Failed attempts remain. [Partial resource accounting](resource-use.json) charges each measured process once, preserves all six original Gray Go reservations and leaves control/GPU commitments unchanged. Some preliminary verification overhead remains unmetered.

Full source admission, independent Retrace sequence evidence, fresh complete pre/post prices, matched controls, the full fixed-seed target comparison and production entry remain required. No full new cohort, GPU work, pilot or study is selected.
