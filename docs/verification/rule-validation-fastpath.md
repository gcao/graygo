# Canonical integer validation fast path

The full suite passes **135 tests in 42.403 seconds**. Raw output and machine/source metadata are preserved in `rule-validation-fastpath/`.

Profiling a real 512-start compact replay batch found repeated `numbers.Integral` abstract-base-class checks dominant. Board/history colors are already canonical Python integers after packet parsing and reference resolution. The shared normalizer now validates ordinary ints directly, preserving NumPy/other Integral normalization to plain int, rejection of booleans/nonintegral/out-of-range values, historical copying and all resolved-state group/history/pass checks. It does not bypass validation or change rules.

Regression fixtures cover native/NumPy integers, invalid scalar types/ranges and dead full boards. Existing codec round trips, differential reference observations, actual neural-update equality and complete mid-episode trainer continuation still pass. Runtime improvements require the paired follow-up measurements; no throughput claim follows solely from the hotspot change.

The same suite extends the independent matrix auditor to RPS neural/tabular traces, including reservoir replacement/replay wraparound and corrupt-report rejection. Three-action uniform probabilities exposed a float32 reconstruction ordering difference: division must precede epsilon multiplication as in the actual sampler. The auditor now reproduces that numerical definition exactly; security-gap calculations, gate tolerances and all training data remain unchanged.
