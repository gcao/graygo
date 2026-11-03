# Proposed completion of original table controls

**Unselected. Owner approval is required before the control ceiling changes or any case below launches.** The immutable [full proposal](../../experiments/nfsp-control-completion-scope-proposal.json) contains the exact sources, original seeds, full primary recipes, raw funding-profile hashes, independent-audit bindings and complete process caps.

| Fixed group | Original seeds | CPU ceiling per complete 15M case | Group maximum |
|---|---|---:|---:|
| Primary branching table R4 | 0, 1, 2 | 4.25 hours | 12.75 hours |
| Primary gray-pure table R4 | 0, 1, 2 | 4.5 hours | 13.5 hours |
| Primary gray-mixed table R4 | 0, 1, 2 | 4.5 hours | 13.5 hours |
| Primary delayed table R4 | 1, 2 | 5.5 hours | 11 hours |
| Same-seed delayed table R1 comparison | 1, 2 | 4.25 hours | 8.5 hours |
| **13 new cases / 195M aggregate records** | | | **59.25 hours** |

Existing recorded use plus retained unknown overhead is **11.732245 CPU process hours**. Reserving all thirteen named complete processes gives **70.982245 hours**. The proposed aggregate ceiling is therefore **72 hours total**, including all earlier costs, with about 1.02 hours remaining. This is a ceiling, not a wall-time promise or a fresh seventy-two-hour grant per case. Costs sum process elapsed across workers; parallelism can reduce calendar time, without reducing the accounting total. The original twofold margins and upward rounding remain; partial-buffer/scaled startup/evaluation/checkpoint forecast limits remain explicit.

The original [design §6.2](../../design.md#62-a2-work-allocations) says: “CPU ceilings remain four elapsed hours for resolver/evaluator work and twelve for tabular controls.” The thirteen full controls cannot fit the 963.918222 seconds still unallocated under that ceiling. The default **43,200-second limit remains unchanged**. No proposal field or numerical assessment grants execution authority. Approval must be a separate owner decision bound to this exact proposal hash, 72-hour limit and thirteen-case list; tests reject absent, mismatched or altered decisions. Existing resource units and prior costs are preserved.

All eleven missing primary table slots and the two same-seed R1 comparison controls are included, with fresh independent initialization/RNG/codebooks and the unchanged 15M caps, 500k reads, opening/ramp/epsilon/target clocks and numerical gates. Existing completed controls and their misses remain. Optional extensions, additional fixtures/cases and any further cost require a separate within-budget selection. The proposal does not select neural training, a CUDA variant, a 9×9 pilot, project/study scope or production work.

The builder freshly validates all sixteen applicable CPU profile audits and full original inputs, committed audit producers/allocations and exact live CPU/dependency settings. GPU diagnostic failures stay visible and that variant remains unqualified. [Job](job.json), [retention](retention.json) and [resource use](resource-use.json) retain the read-only proposal provenance and cost. A2 remains fourteen completed cases; the two original delayed neural workers continue under their already selected bounds.
