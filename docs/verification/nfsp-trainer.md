# NFSP matrix trainer verification

The full suite passed **114 tests in 22.907 seconds** on `ai`; raw output and source/hardware manifest are in `nfsp-trainer/`. This is trainer verification, not formal NFSP convergence evidence.

Neural and tabular runs match uninterrupted training exactly after pausing at 118 records, inside a 16-record update interval: future Q/target/average weights, replay/reservoir contents, RNG streams, complete episode logs and exact evaluation curves agree. Complete runs retain one verified resume bundle and permanent evaluation weights. Closed reservoirs do not produce labels or backlogged supervised updates. Budget stops preserve recoverable state, and changed plans/sources cannot silently resume a lineage.

The tabular control is a real Q learner using the shared replay/optimizer path. Its published policy equals an independent full-reservoir average, including replacement and tied labels. Integer tie-cardinality histograms allow bit-exact reconstruction. A controlled unique-best-action pure82 fixture preserves identical exploratory behavior across greedy/executed-label arms while showing contamination only in executed-action averaging. It is a diagnostic, not the design's formal 15M pure-control result.

Shared inverse-CDF sampling agrees for scalar and batched calls and never chooses zero-mass actions. The installed CLI successfully runs and resumes a completed tabular case and writes a report with the actual matrix schema. A stale writer cannot overwrite newer progress. Committed audit-prefix corruption fails; an uncommitted crash tail is removed on explicit resume without changing the committed prefix.

The default primary protocol is 15M aggregate own-side records (7.5M per role), exact evaluations every 500k records, 128-target Q/SL cadence, 4,096-record Q warm-up, 200k replay/reservoir, R=1, 12,800-record refresh, Adam 3e-4, and fixed 5M/7M/7.5M opening/ramp/exploration clocks. The learned average policy is graded on original-game gaps at the final five evaluations. Terminal matrix profiles cannot price multiround target inference or 9×9 learning; those profiles and all remaining A2 gates are still required.
