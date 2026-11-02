# Baseline-only paired variance pilot

Source `ec72b75888db2bd59a4dc6370127d5bb16bca6e3` ran the frozen tactical subject and random incumbent against frozen random/tactical opponents from official empty 9×9 starts. The complete confidence family and both 32/100-game planned looks were frozen before collection. Only the initial 32-game-per-pairing look was collected: 16 independent color pairs per opponent, four games per pair including the matched incumbent, 128 physical games overall.

| Opponent | Mean subject-minus-incumbent utility | Sample variance of pair differences | Simultaneous lower bound |
|---|---:|---:|---:|
| random | 0.937500 | 0.462500 | -2.000000 |
| tactical | 1.281250 | 0.398958 | -2.000000 |

The finite pilot completed in 41.683 CPU process-wall seconds; the original process is terminal. SQL committed-pair checksums, unit IDs, opening roots, score/utility signs and aggregates validate. Both confidence bounds were recomputed independently, and paired-unit IDs align across opponents for later covariance analysis. JSONL exports retain every game receipt and legitimate repeat; original SQL evidence remains on `ai`.

The small sample has deliberately wide bounds. It exercises actual evidence collection and paired variance, not a practically-useful-improvement power claim. It does not establish the promotion gate's 80% power or <=5% false-promotion target across all required scenarios, does not calibrate a future protected opponent set, and authorizes no candidate/pilot training. A whole-gate calibration using separately designated agents and its fixed protected pool remains required before promotion.

The delayed R1 learning screen remains independent under its original PID 2217027; evaluation did not feed its replay, reservoir or opponent population.
