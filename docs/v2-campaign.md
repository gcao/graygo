# V2 training handoff — November 10, 2026

This is a separate, unlimited v2 campaign. It preserves pretrained knowledge,
starts fresh rule-versioned buffers, and evaluates frozen candidates on CPUs while
the GPU learner keeps advancing. V1 checkpoints and evidence remain available.

## Initialization and observations

- Learner: the sealed 80,478,208-record Q and average networks. Its latest v1 gate
  scored 57.925% against the incumbent; that is initialization evidence, not a v2
  strength certificate.
- Initial incumbent: the 52,690,944-record champion. Historical references start
  from 24,641,536, 34,865,152, and 39,583,744-record average weights. All comparisons
  collect fresh games under v2; v1 match results are never reused as v2 evidence.
- 64 channels, eight residual blocks, 14 globals. Two additional spatial channels
  bring the input to 32 planes. Old spatial coefficients stay in their positions;
  old global coefficients move past the added channels, whose weights start at
  zero. Q and average each retain their trained parameters. Optimizers and target
  bookkeeping start fresh; the target network starts from Q.
- The extra planes mark our and their currently available placements that would
  repeat a prior board **if the other player passed**. They are conditional hints,
  not individual legality masks. Simultaneous play can make such a move productive.
  The full board ledger remains authoritative and the observation remains partial.
- Hints are computed once per immutable state using the single-placement group
  analysis, packed into the observation packet, and transformed with the board.
  Replay decoding uses the existing checksummed C++ base decoder plus vectorized
  bit unpacking for the two hint planes. It never reconstructs game history.

Both replay and the supervised reservoir have capacity 500,000 and begin empty.
The reservoir remains uniform over eligible **v2** labels; recency weighting is
not silently introduced. The pretrained average is immediately usable. The
exploration clock retains the parent's 80.48M offset, giving epsilon 0.01, while
fresh record/update counters start at zero. There is no repeated five-million
record from-scratch opening. Average-mode training remains the raw-average
control; release selection remains temperature-zero placements with learned PASS.

## Gate and scheduling

`tools/v2_gate.py` uses a fixed mixture of nonnegative betting processes for
bounded color-pair observations. Each factor is `1 + b * (X - threshold)` with
stakes fixed before collection and small enough to stay positive on the declared
support. Under the null, the mixture is a nonnegative supermartingale, so Ville's
inequality controls repeated looks. See
[Waudby-Smith and Ramdas](https://arxiv.org/abs/2010.09686).

All promotion conditions must pass: positive head-to-head utility, historical
utility differences above -0.15, and historical absolute utility above -0.10.
The conditions form an intersection-union test: if a required condition is false,
false promotion requires crossing its valid test threshold. Each required test
may therefore use the full attempt alpha. Diagnostics do not consume it.

Attempt i receives `0.05 / (i * (i + 1))`; the sum over all actual future attempts
is 0.05. Screens, checkpoints, and skipped milestones do not consume attempts.
A crashed confirmation resumes the same attempt, policies, seed and evidence.
Color pairs remain the independent units, retaining within-pair dependence and
counting draws. Registered looks are 500, 2,000 and 4,000 games per opponent.
There is no positive pooled-gain promotion requirement.

Before launch, `calibrate_v2_gate.py` must pass power and null checks and its code,
source and gate-plan fingerprints are bound into the campaign plan. At 4,000 games,
10,000 trials per case fitted to the retained paired evidence gave approximately
99.9%, 97.8% and 83.3% joint power on actual attempts 1, 10 and 100 for a true 55%
head score and unchanged historical strength. The tested head and historical
boundary-null false-promotion rates were below the allocated alpha. These are
conditional power estimates under the fitted outcome model; universal error
control comes from the betting/IUT construction, not from these simulations.
Observed v2 variance and power must continue to be monitored.

A frozen CPU job first screens 100 games per opponent. A head score of at least
57% or the two-hour confirmation timer starts a full gate with fresh seeds.
The screen interval is 15 minutes at 60%+, 30 minutes at 55%+, and otherwise one
hour. One job is pending at a time. Learning continues throughout. Results bind
candidate, incumbent, source, rules, attempt and raw paired evidence. A stale
incumbent result cannot publish. Rejection and promotion never reset the learner.

The learner checkpoints and archives every 262,144 fresh records. The existing
atomic store retains two heavy resume generations and preserves small immutable
weight archives. The twenty-GiB disk reserve remains in force.

## Runtime and publication

Use `tools/v2_campaign.py --prepare ... --calibration ...`, then the same pinned
source with `--run --run-dir ...`. The source, tools, decoder and initialization
hashes are immutable for a run. A bounded `--no-publication --until-records ...` smoke run can validate the actual campaign and checkpoint resume before the service handoff; it cannot publish or run without a finite validation endpoint. SIGTERM saves at a balanced record boundary.
Systemd restarts the controller after failures; pending evaluations keep their
identities and resume their committed paired prefixes.

The GPU run uses 64 environments and native decoding. An 8,192-record profile
measured 340 records/s with Python decoding and 16 environments, 618 with native
and 16, and 858 with native and 64. The Python/native 16-environment runs produced
bit-identical trained Q and average tensors. These short profiles include a new
buffer's learning start; they are not sustained-campaign throughput guarantees.

V2 releases go to the separate `v2-champion.json` pointer. The arena chooses it
for v2 loop-champion games and continues using `champion.json` for v1 games.
The manual candidate catalog is independent. Full public board history reaches
the local inference worker so promoted 32-plane models can compute their hints.
Historical 30-plane models continue to use their original observation shape.

The existing hourly timer and `runs/training-watch.log` continue to work. The
observer reports fresh v2 records, the warm-start record, hints/native settings,
rejection fraction, CPU job progress and the last gate decision. It uses no Torch
or training operations. For example:

```sh
ssh ai 'tail -f ~/proj/graygo/runs/training-watch.log'
```

## Controlled follow-up experiments

Keep this first v2 campaign as the measured baseline. Compare release-aligned
average-mode exposure, a recent/history reservoir mixture, and auxiliary final
score prediction in separate arms from a common sealed v2 checkpoint and matched
compute. Do not combine them into one unexplained algorithm change. The primary
reward remains win/draw/loss, and any score target is auxiliary. Historical data
age alone is not a reason to discard all historical labels.

The baseline must first accumulate genuine v2 outcomes. These follow-up changes
are not enabled by this handoff, and their benefit is not claimed by the runtime
or gate calibration tests.
