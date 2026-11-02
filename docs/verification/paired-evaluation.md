# Paired evaluation and milestone-gate verification

The full suite passed **160 tests in 50.078 seconds**; raw output and source/machine metadata are preserved in `paired-evaluation/`.

## Sampling and durable evidence

Version 2 exchanges the agents' colors on the same complete root in each independent unit. Actor/action RNG streams are private; both moves are committed from the same immutable state. Candidate/incumbent comparisons share openings and action seeds against the same frozen opponent. Episode mixtures choose privately once per game. Complete-root openings are declared new episodes; actual prefix-continuation mixtures remain the responsibility of the history-conditioned solver.

Policy weights/representations, actual rules, opening catalog, confidence-family hash, sample looks and utility/improvement endpoint are frozen before collection. Archived neural weights verify checksum/configuration and retain training-run/seed/record provenance. A diagnostic model cannot become a 9×9 milestone checkpoint merely by naming it so. SQL transactions commit whole paired units, with deterministic unit identities, payload checksums, scored outcomes and trajectory fingerprints. Resume reuses committed units exactly; missing/corrupt evidence and changed plans/policies/sources fail. Repeated trajectories are retained and reported, never deduplicated.

## Confidence procedure

For n independent paired units in support [a,b], the one-sided lower bound is the sample mean minus `sqrt(2*s²*log(2/delta)/n) + 7*(b-a)*log(2/delta)/(3*(n-1))`, clipped to a. Here s² is unbiased sample variance. This is the scaled empirical Bernstein bound from [Maurer and Pontil (2009), Theorem 4](https://arxiv.org/abs/0907.3740), applied to reflected observations for a lower bound. Alpha is allocated over every endpoint and planned look; arbitrary dependence between those tests is allowed. Independence is required between paired units, not between the games within a pair. Confidence statements remain conditional on the tested frozen models/distribution; games do not erase training-seed uncertainty.

The method is conservative. Protected-pool promotion still requires the design's separately designated pilot outcomes, joint false-promotion/power calibration, fixed weights and numerical gate. The CLI explicitly does not promote a policy. This layer does not claim that a +0.03 scenario has 80% power.

## Milestones

The pilot protocol registers all development runs 100/101/102, both 4.5M/4.75M/5M and 6M/7M/8M Q windows, both opponents and 500/2,000-game looks. Including post-opening average diagnostics yields 108 one-sided comparisons. Snapshot passes require both baseline checks at that same checkpoint; two such passes are required per fixed window for every seed. Average diagnostics must be present before cohort completion. The latest completed planned look is used; hidden later looks, missing snapshots, substituted weights/representations, altered thresholds and moved clocks fail.

Development opening is permitted at 5M only after its pre-window clears. Later insertion requires the persisted fixed-boundary opening event. Confirmation seeds 200/201/202 follow the fixed schedule regardless of performance reads. Screening grades the deployed average at 20M: lower bound >0 versus random and >=+0.10 versus tactical, for all three seeds. These decisions remain local milestones, not A2 or promotion certificates.

Tests include exact small Bernoulli coverage enumeration, independent bound arithmetic/scaling, paired-color/episode sampling, identical-incumbent zero deltas, checksum/plan corruption, repeated-trajectory retention, actual archive provenance, SQL resume and the installed CLI. Pilot/screening fixtures enforce full cohorts, same-snapshot opponent clears, opening records and immutable serialized protocols. No large-board learner or scientific endpoint has passed solely because these implementation tests pass.


## Version-2 asymmetric-opening correction

The earlier combined board/agent color swap preserved a nonempty opening's advantage for the same agent. A regression with identical pass policies on an asymmetric root exposed paired utility +1 instead of zero. The corrected pair retains the exact public board and exchanges agent roles; utilities +1/-1 cancel. Official empty openings are unchanged. Old version-1 nonempty data requires its original interpretation and cannot be resumed/loaded as version 2. See `scientific-contrast.md` for the new matched multi-rule collector.
