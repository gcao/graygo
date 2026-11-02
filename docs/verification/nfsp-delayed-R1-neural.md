# Independently audited delayed R1 first screen: complete

Both seed-0 trials completed their original **15M-record** allocations on pinned source `9fefeee` / SHA `ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e`. Original driver PID 2217027 is absent and its terminal state records both cases complete. It must not be relaunched as a live continuation.

| Complete final-five maximum | Tabular control | Neural learner | Requirement |
|---|---:|---:|---:|
| Supported early/root Q error | 1.1583649069 | 1.9562109606 | <=0.10 |
| Greedy-Q original-game gap | 0 | 0 | <=0.05 |
| Deployed-average original-game gap | 0.6318537859 | 0.0000026410 | <=0.10 for the numerical chain band |
| Actual CPU-process seconds | 7,218.411 | 7,354.825 | Original declared case ceilings |

The independent [neural audit](nfsp-delayed-R1-neural/audit.json) recomputes every scheduled permanent model archive and verifies the original-game values/gaps, native opponent mixture, opaque states, episode/coverage/mode/label clocks, Algorithm-R selections, private/replay/supervised RNG continuations, optimizer/target counters and retained transitions. The [tabular audit](nfsp-delayed-R1-control.md) remains separate evidence.

**Credit and averaging are separate.** Both greedy-Q policies clear the root gap threshold, but both Q estimators miss supported value accuracy. The neural average clears its numerical chain gap band; that success does not establish delayed credit or pilot eligibility. The trainer's composite `average_readiness=not_ready` includes prerequisites, so it must not be described as a neural averaging failure. The tabular average remains above the band and is still improving at the cap. These results do not diagnose a neural bug or prove that more compute repairs the route.

Each trial has 7.5M own-side records per role, 12M ordinary records, 29,289 Q updates, 15,621 average publications/updates and 1,171 hard target refreshes. Both have 87,500 complete episodes and 811,200 eligible labels. The full original final-five window is used; no early stop, changed opening, seed filtering, label reset or marginal extension was selected.

The design's next comparison is one-step R4 at the **same 15M records**, with the same source, seed/codebook, coverage, 5M opening, 5M-7M ramp, 7.5M A2 epsilon clock, batch 512 and 12,800-fresh-record target refresh. Its measured matching profiles already exist. This selects only a paired seed-0 tabular/neural first screen; all later cohorts, target selection, full campaign cost and A2 exit remain open. See the [funded plan](../experiments/nfsp-delayed-R4-primary.md).
