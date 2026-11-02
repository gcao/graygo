# Native rule decoder verification

The full target-machine suite passed **142 tests in 48.111 seconds**. Raw output, machine/source data, immutable runtime build and wheel evidence are preserved in `native-rule-decoder/`.

## Exact scope

The optional C++17 backend expands and validates complete `graygo/rule-state/1` packets into the same 30 spatial channels, 14 scalar inputs and legal masks as the independent Python reference. It supports sizes 3–19 and 32-bit round limits. It does not resolve moves, score games, supply training targets or access an opponent's commitment. Unsupported configurations and malformed states fail; no implicit reference fallback changes a selected backend.

Native validation checks rule/version fingerprints, packet length, board/restriction padding, missing-history zeros, move ranges, history/round counts, trailing mutual-pass count, resolved-group liberties and terminal legality. Group/liberty generation uses the same four-neighbor torus. Explicit sequence-padding rows produce zero features/masks without pretending to be valid states. History slots, relative colors, historical moves/pass flags and countdown thresholds preserve the reference semantics.

Differential checks cover thousands of generated observations across 3×3/9×9/13×13/19×19, both collision rules, both perspectives, terminal modes, geometric/color transforms, independent restrictions and wide counters. Mutated packets have matching acceptance/rejection and observations. Malformed/dead states remain rejected. Complete bounded neural Q/target/average weights, metrics and trajectory audit bytes remain bit-exact through active-episode checkpoint continuation using the native decoder.

## Build and identity

`graygo build-native` compiles owned source into an immutable content-addressed directory and atomically publishes its library plus manifest. Compiler identity, flags, source checksum, ABI and library SHA-256 are recorded; fast-math is not used. Existing libraries are verified and reused, never replaced in place. A selected native descriptor requires the exact checksum; missing, changed or unsupported libraries cannot silently resume. Native source participates in run source fingerprints and is included in the distributable wheel. The current builder is Unix/C++17; the Python reference remains available independently.

Native selection is explicit in buffer schemas, trainer plans and cost budgets. The recorded library must remain available for restore, just as the declared Python/Torch environment does. Performance, full-capacity IO and complete CUDA learner costs still require measurement; feature correctness alone does not price a campaign.
