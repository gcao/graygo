# Protected-pool promotion statistics and calibration

The full suite passed **170 tests in 59.485 seconds**. Raw output and machine/source metadata are in `protected-promotion/`. This verifies the procedure; no current 9×9 candidate is promoted or statistically qualified by these tests.

## Frozen comparison

`ProtectedPool` caps the initial set at ten distinct versioned policy identities and freezes weights, rules and opening distribution. `PromotionProtocol` freezes 500/2,000 looks or the explicitly predeclared 8,000-game extension, simultaneous 95% coverage, +0.03 useful improvement, -0.15 per-opponent regression tolerance and -0.10 absolute floor. Changes produce a new protocol/family identity; serialized tolerances cannot quietly change.

The numerical gate forms each independent unit's weighted candidate-minus-incumbent improvement before estimating variance. This retains cross-opponent opening/color covariance. It jointly requires aggregate lower bound >0, every regression lower bound >=-0.15 and every candidate-utility lower bound >=-0.10. Actual scored pair evidence must have aligned unit/opening identities, the same frozen candidate/incumbent and the complete protected set. Pool weights are normalized before collection; only floating-point excursions outside the mathematically bounded convex support are clipped.

A candidate's collection plan must bind both the whole protected protocol and the exact pre-candidate calibration-report hash. A later completed look cannot be hidden. Statistical qualification also requires independent calibration agents; renaming identical frozen weights or baseline mechanisms does not create independence. Display labels are outside policy identity. Actual promotion additionally requires the project controller's verified A2 and 9×9-smoke receipt, bound to the candidate/incumbent identities; the statistical module does not manufacture that prerequisite evidence. This global controller integration remains required work.

## Whole-gate calibration

Calibration loads actual checksummed SQL pair evidence read-only. It samples entire cross-opponent paired opening/color rows, retaining incumbent outcomes and joint opening dependence. Declared counterfactual candidate means use a recorded mixture with +/-1 outcomes, driven by common cross-opponent uniforms. The resulting scenario model changes draw/variance structure as stated; its power is conditional on that pilot-derived model, not a transferable guarantee for arbitrary future models.

Required cases include zero improvement, uniform +0.03, nontransitive gains/losses, a regression at and beyond -0.15 in each protected matchup, an absolute-floor violation in each matchup, and missing A2/smoke prerequisites. The same numerical/finite-confidence implementation is evaluated at every planned look, stopping only when the complete gate clears. Infeasible scenario means remain infeasible; they are not replaced by an easier task or a relaxed floor.

Monte Carlo pass probabilities have their own simultaneous finite-sample uncertainty. Qualification requires the useful scenario's lower power bound >=0.80 and every violated criterion scenario's upper false-promotion bound <=0.05, complete scenarios, actual separately designated pilot evidence and declared minimum sample/replication counts. Version-one defaults require 250 paired pilot units and 1,000 simulations; synthetic fixtures cannot qualify a real gate. Missing/infeasible/underpowered evidence returns `not_calibrated`.

Stateless trial seeds, committed progress and a cumulative time clock reproduce interrupted simulations without replaying completed work. CLI plans bind protocol, pilot hashes, source, simulations, seed and total ceiling; a whole-run lock prevents overlapping writers. Reports bind their full content hash and qualification rechecks counters/bounds. The CLI performs calibration analysis only and does not update an incumbent.

## Checks and limits

Fixtures independently isolate aggregate failure, a protected regression and an absolute-floor failure. Paired covariance, ten-opponent bounds, immutable looks/tolerances, real SQL alignment, display-label identity, incomplete evidence, infeasible +0.03 scenarios, synthetic-pilot exclusion, report corruption, deterministic resume and CLI refusal/resume all pass.

The older 16-pair baseline pilot is a pipeline/variance demonstration, not enough to qualify a promotion round. Real pre-candidate calibration, protected-pool freeze, applicable sample/evaluation budget, A2 approval and production 9×9 readiness remain necessary. No game count removes training-seed uncertainty or establishes full-game exploitability.
