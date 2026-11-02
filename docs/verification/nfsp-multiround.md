# Multiround NFSP trainer verification

The final-source suite passed **122 tests in 36.991 seconds**. Complete output and machine/source metadata are preserved in `nfsp-multiround/`.

Neural and tabular delayed runs, plus neural and tabular reachable mixed Gray Go runs, exactly reproduce future online/target/average weights, replay/reservoir arrays, coverage/overlap metrics, private RNG streams, episode audit bytes and exact evaluation curves after a mid-episode checkpoint. The delayed pause occurs after Q has updated while the active actor still uses older frozen parameters. A final record limit inside a continuing game preserves a nonterminal replay successor and active state; restored training does not invent an ending.

The analytic delayed evaluator recovers hand-set exact values on both branches at multiple horizons and roles. It catches a deliberately wrong avoided root-action value even when the favorable action remains greedy. A branching fixture independently recovers uniform-continuation root action values `(-0.125,-0.3125)` and exact game values `-1/7` / role-swapped `+1/7`. The bounded mixed adapter replays the actual fixture prefix and reproduces its complete public root and value `-1/4`; the pure bounded root has value zero.

A complete supported-coverage macrocycle verifies 8/1/1 blocks, per-role fresh counts, supplemental root edges, separate ordinary counters and an R=4 target refresh driven by fresh records. Replay starts and overlap diagnostics agree with actual loss-bearing counts. An independent overlap fixture checks `min(1,pi/mu)` using the logged conditional action probability, without multiplying the episode-mode probability a second time. The installed CLI writes actual 3×3/6-round research rules and resumes an active mixed-Go episode.

Full 15M A2 allocations require declared cost-profile evidence. Diagnostic purpose, shortened budgets and changed architectures cannot pass the formal flag. Implemented gates still need actual convergence experiments; this report does not establish delayed credit, deployment readiness, any 9×9 pilot, or the scientific endpoints. CPU exact-resume evidence does not establish bit-exact CUDA continuation; runtime determinism settings are explicitly recorded for later accelerator verification.

The Q continuation reference in this historical implementation report is superseded by [the native-opponent correction](nfsp-native-reference.md). Training and resume mechanics are unchanged; complete current evidence is recorded there.
