# Confirmed bias from supplemental opponent intervention

## Exact reproduction

The [stationary diagnostic](nfsp-forced-opponent-diagnostic/README.md) initializes every delayed-chain Q value exactly, uses forty equivalent continuation actions and exercises the original frozen `9fefeee` collector, replay sampler, expected one-step target and Q loss. For each controller orientation it collects eight ordinary deterministic favorable-root episodes, one forced favorable root and one forced unfavorable root, followed by a normal suffix check. It uses 44 fixture records total and is not a learner-selection experiment.

| Passive-root quantity | Native objective | Original 8/1/1 replay mixture |
|---|---:|---:|
| Exact stationary value, epsilon zero | -1 | -0.8 |
| Initial exact Q | -1 | -1 |
| Q after one demonstration SGD update | remains correct if ordinary-only | -0.96 |

The passive player has just one root action, with own behavior/target probability one. Its opponent's branch choice was forcibly changed in the supplemental episode. That changes the passive role's transition kernel; it is not ordinary off-policy sampling of that role's own action. The collector nevertheless marks both forced-root records `value_mask=true`, replay samples every start and the shared Q loss uses every selected row. Thus an exact native Q value receives a nonzero gradient toward the wrong coverage-mixture value even with perfect continuation values and a frozen target. More reuse or faster propagation does not remove that stationary root bias.

The forcing player's two conditional root-action targets remain exactly +1 and -1. Its own forced action is a legitimate off-policy action-value sample. Both roles' normally generated branch suffixes also remain valid. Supplemental records still belong outside the averaging reservoir.

## Size at the primary exploration floor

With eta=1/10, epsilon=1/100 and an optimal average policy, the native opponent chooses its favorable root action with probability 1999/2000. The passive native value is -999/1000. Mixing eighty percent ordinary episodes with balanced ten-percent favorable/unfavorable forced episodes gives -999/1250 instead, a bias of **999/5000 = 0.1998**. This alone exceeds the unchanged 0.10 supported-value tolerance. Own-action importance correction cannot repair it at the passive root, where both probabilities are one.

At the inspected R4 neural 11.5M point, the fifty-/hundred-round passive roots predict about -0.74 to -0.76 against native values near -0.999. This is consistent with the diagnosed bias plus approximation/nonstationarity. A separate two-hundred-round unfavorable-branch collapse has error around 1.91 and is **not** explained by the stationary root calculation. The diagnostic does not claim to explain every credit miss or guarantee convergence after correction.

## Required implementation correction

The design permits legitimate forced-root **own-action** samples. That permission does not make the other role's opponent-intervened root transition a native-opponent Q sample. Keep the forcing role's conditional root edge and both normal suffixes; exclude the passive pre-intervention root from Q targets unless a separately justified opponent-distribution correction is implemented. Do not reveal a forced outcome/origin flag to the actor to make the gate easier, remove the supported root-action requirement, or change the native reference to the coverage mixture.

The correction must use the shared production replay/loss/cadence path and make value eligibility effective. All collected own-side records, including excluded-target records, still count once toward the fixed aggregate fresh/time budget. Q optimizer reuse and the 12,800-fresh-Q target clock must use actual Q-eligible records, while epsilon, opening and the 5M-7M hand-off retain the aggregate record clock. Preserve separate ordinary/label/per-role/eligible counts and exact resume across a Q boundary that can fall between the two serialized own-side records of a simultaneously committed round. Commitments and frozen labels must still precede learner updates.

Acceptance should include both stationary role orientations, valid own forced-action/suffix targets, no supplemental averaging labels, replay-start filtering, exact counters/refreshes, per-role coverage, and interrupted continuation. Corrected source then needs matched profiling and a separately recorded corrective allocation before any new full training. The original profiles cannot silently fund changed collection/cadence work.

## Existing trial interpretation and scope

The original R1 trials and completed R4 control retain their exact archived numerical results and independent storage/RNG/value audits. Those audits do not prove this intervention kernel is the native one. Their credit misses are now explicitly confounded by this confirmed collector/eligibility defect; they cannot cleanly select against one-step propagation or justify jumping to Retrace first. Unaffected ordinary-only short matrix evidence remains separately scoped. Correction does not promise that every other approximation/propagation problem disappears.

Original R4 driver PID 2320416 continues its unchanged full matching neural allocation; it is neither restarted nor edited. Its complete numerical results remain diagnostic evidence under the original source. No large corrective run, target change, stability extension, 9x9 pilot or scientific campaign is selected by this report. All original thresholds, full objective and project resource limits remain intact.

The controlled diagnostic interval took **0.680 CPU process-wall seconds**, excluding initial interpreter/import/manifest setup. It uses oracle initialization deliberately to isolate the update defect and does not claim a trained policy or an implemented repair. The [machine-readable report](nfsp-forced-opponent-diagnostic/report.json) binds the exact production source, driver and both fixtures.

Reproduce with the original package on `PYTHONPATH` and a fresh output path:

```sh
PYTHONPATH=/home/gcao/proj/graygo/.worktrees/nfsp-tasks-9fefeee/src .venv/bin/python tools/diagnose_forced_opponent.py runs/forced-opponent-diagnostic-repeat
```


## Correction status

The [shared correction](nfsp-q-eligibility.md) is now implemented: passive pre-forcing targets are masked, Q sampling/losses enforce eligibility, aggregate and eligible freshness are separate, and task updates respect exact eligible boundaries after both commitments are fixed. The stationary regression no longer pushes exact native Q away from its correct value. Exact CPU/CUDA odd-clock continuation and independent corrected-run audits pass. Original diagnostic artifacts/source are unchanged. Full corrected convergence and matched resource pricing remain unproven; no new long training or target variant is selected by this implementation status.
