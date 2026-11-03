# Original A2 to optimized-runtime source comparison

The original corrected A2 package is `db9db2e...` at frozen `ca45a53`. The current package is `a522fe2...`, unchanged since `c3b8b5f`. [Inventory](inventory.json) pins both full package hashes and follows static local imports from `nfsp_training`, `task_training` and package initialization, including conditional/function-local imports. Compared source is parsed as data and never imported or executed. External dependencies and dynamic import/execution sites remain visible.

There are **24 byte-identical dependency modules, including package initialization** (23 excluding it). These include both trainers, production NFSP losses/targets/masks, model definitions, buffers, schedules, task adapters and task/evaluation code. The original closure contains 27 modules and the candidate 28. Three existing modules differ and one dependency is new:

| Module | Review boundary |
|---|---|
| `engine` | Prepared single-placement resolution and preserved literal reference resolver. Existing finite differential and complete-game evidence is retained under [prepared placement](../prepared-placement/README.md). |
| `agents` | Tactical ranking via the prepared resolver; priorities, seeded choices and RNG behavior have the separately recorded differential checks. |
| `nfsp_collection` | Optional frozen-forward pool, routed Q/average calls, restore metadata and cache invalidation. A2 callers default to no pool; that observation alone is not a complete old/current trajectory proof. |
| `frozen_forward` | New opt-in CUDA capture executor. Current eager/graph exact outputs and resumed CLI trajectory equality are separately retained under [CUDA forward verification](../frozen-cuda-forward/README.md). |

The comparison also lists changed files outside the static task-import closure. Their exclusion does not certify that arbitrary runtime hooks, dynamic imports, settings or environments cannot affect a task. Source identity, behavior/configuration, numerical runtime and original cohort evidence remain distinct requirements.

**Compatibility remains unqualified.** This inventory does not rewrite any older run, loosen the A2 collector's exact-package checks, merge cohorts, mark missing seeds complete, or grant training. Next verify actual changed call boundaries and applicable old/current trajectories/settings, then explicitly connect that proof to project entry. The [validation record](validation.json) retains the real CLI pins and 24 passing focused tests (eight new source-inventory tests plus sixteen existing resource checks). The unchanged application's latest complete suite remains 430 tests; no new full-suite run is claimed.

```sh
.venv/bin/python -m tools.task_source_diff \
  --reference .worktrees/nfsp-eligibility-ca45a53/src/graygo \
  --candidate src/graygo \
  --expected-reference db9db2eb532d0a2782486a486743701a0a1914277616ff5d5c0406f3b6501873 \
  --expected-candidate a522fe224ccf6db1c528d3fea1c896869038b29e48dbda9e00174ca8116a24df \
  --output /tmp/graygo-task-source-diff.json
```

An existing output is preserved. Both source trees are checked again after analysis, and output inside either compared package is rejected.

The [trained default-actor boundary](actor-boundary/README.md) now passes exact old/current checks for both full original 15M archives, all 2,788 observations and explicit private sampling/restore partitions. Full learner-update/buffer trajectory checks remain separate; a newly declared bounded matched pipeline diagnostic will test those next.
