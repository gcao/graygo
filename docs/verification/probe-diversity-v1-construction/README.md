# Terminal multigroup construction output

Original construction PID 2384063 is terminal/absent. Its frozen source is `7da67d9`, package SHA `8376aead9575a4ff63ec35c1f78ff0ab1ca847c1f1f6b4218ad8e036463b505b`, retained in `/home/gcao/proj/graygo/.worktrees/probes-multigroup-7da67d9` on `ai`.

The original run at `/home/gcao/proj/graygo/runs/collision-probe-diverse-7da67d9` stopped at its 1,800-second cap after 150,507 proposals, with 24 distinct complete-game families: twenty removal/four recoloring pairs, twelve increasing/twelve decreasing orientations, fifteen with gray-containing roots and fourteen with mixed-equilibrium witnesses. Actual construction time is 1,800.007 process-wall seconds; the small overrun is one bounded attempt.

This is below the required 32-family coverage. No `benchmark.json` was published. All 24 partial exact proofs and actual 199-round setup traces remain inside the copied `report.json`. No trained policy was evaluated and no scientific conclusion is selected.

The separately allocated independent exact-proof/source-seed audit is in progress at `/home/gcao/proj/graygo/runs/collision-probe-diverse-7da67d9-audit`, original audit PID 2393525. The frozen auditor is `runs/collision-probe-diverse-7da67d9-audit-driver.py`, also tracked as `tools/audit_probe_construction.py`. A terminal verified audit is still required before calling the full original proposal prefix independently reproduced. Do not restart from a progress/observation timeout; inspect its exact OS handle.
