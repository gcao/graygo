# Three original branching neural cases launched

Original parent 2798293 launched only seeds 0/1/2 as workers 2798312, 2798313 and 2798314. Each uses the unchanged full primary one-step R4 recipe, 15M records, 500k evaluation intervals, 500k buffers and batch 512. Source a522fe2 is imported from its original frozen package; run manifests accurately retain orchestration commit 19b2523 as a separate code context.

The [actual initialization comparison](initial-model-comparison.json) verifies all three private zero-record checkpoints: counters zero, optimizer moments empty, target equal to Q, no active episode or collected roles, and different actual Q and average weights for every seed pair. The [retention manifest](retention.json) seals copied original manifests and complete initial checkpoints. These are not scheduled public evaluation agents at zero.

[The training allocation](../../experiments/nfsp-branching-neural-seeds012-allocation.json) reserves each complete 15,300-second CPU process cap, with graceful budget signalling 30 seconds before the deadline and a hard stop before it. No existing root is resumed or replacement seed started. [The read-only observer allocation](../../experiments/nfsp-branching-neural-seeds012-audit-allocation.json) fixes original process identities, recipes, source plan, auditor, prices and initial seals before final results.

Forty-five million aggregate formal records are selected, not yet completed. No pass, A2 exit, table-control expansion, extension, pilot or study is claimed. CPU neural reservations total 45,900 seconds for this cohort; both existing delayed-neural reservations remain separately funded.
