# Frozen multigroup probe construction allocation

The source/plan allocation is [recorded before launch](collision-probe-diverse-v1-allocation.json). Its [validation receipt](collision-probe-diverse-v1-driver.json) checks the actual frozen package SHA, clean source commit, run/plan hashes, generator, seed, proposal cap and time ceiling. It observed the real OS process at the recorded UTC timestamp; the receipt itself is not ongoing liveness proof.

- Frozen source: `7da67d9e6c803ac5ea0ac8f20b3954dcebbcaa8d`, package SHA `8376aead9575a4ff63ec35c1f78ff0ab1ca847c1f1f6b4218ad8e036463b505b`.
- Worktree on `ai`: `/home/gcao/proj/graygo/.worktrees/probes-multigroup-7da67d9`.
- Original construction PID: **2384063**, now terminal/absent.
- Retained run: `/home/gcao/proj/graygo/runs/collision-probe-diverse-7da67d9`.
- Seed 974214, at most 200,000 proposals and 1,800 construction seconds; stop earlier at 32 distinct classes.
- Independent full-proof/source-seed verification has a separately bounded 3,600-second ceiling. No training or scientific inference campaign is selected.

At initial OS/plan validation the saved progress contained four distinct complete-game families, two with gray roots and one with a mixed equilibrium. This is an initial observation, not a completed 32-family catalog. `progress.json` is updated on accepted pairs, so it may remain unchanged during a live search for another class. Always check the original OS PID, never restart from progress age or a tool observation timeout. A terminal report that remains below 32 classes preserves its partial exact proofs and incomplete coverage.

Once the original construction is terminal, inspect its report and benchmark inventory. If all 32 pairs are present, run the independent proof and source-seed regeneration from the same frozen source:

```sh
cd /home/gcao/proj/graygo/.worktrees/probes-multigroup-7da67d9
PYTHONPATH=/home/gcao/proj/graygo/.worktrees/probes-multigroup-7da67d9/src /home/gcao/proj/graygo/.venv/bin/python -m graygo verify-probes --benchmark /home/gcao/proj/graygo/runs/collision-probe-diverse-7da67d9/benchmark.json --regenerate --time-cap-hours 1
```

Record actual terminal construction/verification time and source-bound artifacts. The ceilings are allocations, not measured consumption. No additional construction run, target relaxation, duplicate-class counting, learned-policy probe evaluation or specialization claim follows automatically from this allocation. The existing full design, independent confirmation cohort and controlled scientific endpoints remain required.


## Terminal construction and independent audit

The original run completed its unchanged 1,800-second allocation after 150,507 proposals with **24 distinct families**, so coverage remains incomplete. The [terminal source-bound output and all partial proofs](../verification/probe-diversity-v1-construction/README.md) are preserved. No full 32-family benchmark was created and no replacement construction or extension is selected.

The separately allocated exact-proof/source-seed audit (original PID **2393525**) is also terminal and verified. It used the same frozen package and committed auditor, validated the exact complete-root/setup/class/near-equilibrium proofs without a floating LP during proof reading, and regenerated the original 150,507-proposal prefix. Its output remains at `/home/gcao/proj/graygo/runs/collision-probe-diverse-7da67d9-audit`. This verifies the original prefix and leaves the 32-family requirement unchanged.


## Independent audit complete

Original audit PID 2393525 is terminal/absent. The [independent receipt](../verification/probe-diversity-v1-audit/README.md) verifies all 24 exact pairs, 48 unique roots/current boards, 9,552 legal setup rounds and the entire original 150,507-proposal prefix, with unchanged input artifacts. It took 1,794.814 measured CPU process-wall seconds; proof reading was 2.587 seconds and regeneration 1,792.223 seconds. The [settlement](collision-probe-diverse-v1-settlement.json) records both measured intervals and unused verification reservation without selecting more work. **Family coverage remains incomplete at 24/32.** No trained policy or gray-specialization result was produced.
