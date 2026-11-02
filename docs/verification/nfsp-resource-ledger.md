# Route ceilings and measured resource accounting

`tools/nfsp_resources.py` implements design §6.4 with explicit units and nullable prices. It separates prospective device-hour ceilings from recorded CPU process-wall measurements and accelerator allocation intervals. It neither qualifies a hardware price independently nor launches or authorizes work.

## Exact route expression

The numerical plan records the selected source, target/reuse and explicit comparison selections. It computes:

```text
N_A2 = 96 + 3*C1 + I4*3*C4 + IT*3*CT + I0*B_0 + E + D
full = P + U + N_A2 + 3*H_D + 12*H_N + 16 + B_U
       + corrective_device_hours + 8*promotion_rounds + other_unreused_device_hours
```

`E` sums three times each selected extension's per-seed ceiling. A protocol cannot receive a second/different-family extension. This is price arithmetic; marginal eligibility and actual extension authority remain separate. The uncorrected diagnostic is at most twelve device-hours and the corrective device reserve at most twenty-four. Optional work is never selected simply because its price is available. Selection flags retain undertaken/committed failed comparisons even if another target/reuse ultimately wins. Other unreused actual costs must remain in their applicable terms; changing a winner does not erase consumption.

**H_D includes H_P**, so the expression does not add `3*H_P` again. H_D must be at least H_P. H_P remains a required input, even though it is not an additional term in the full formula. Its initial per-seed ceiling is twenty-four device-hours. Confirmation contains twelve independent runs, not three development agents reused as confirmation. Scientific evaluation/probe inference adds sixteen device-hours; each selected promotion adds eight. Device-days are hours divided by twenty-four and are not a wall-time promise.

Unknown prices stay JSON `null`; they are never coerced to zero, priced from terminal-only CPU trials, or replaced by the PPO-only total. Explicit zero device-hours can represent CPU-only actual use, with its reference, while CPU consumption remains separate. Required values include P, U, C1, B_U and H_P/H_D/H_N, plus prices for selected optional work. Missing values make the full ledger **uncosted**. An absent project device ceiling also remains explicit. Component-limit violations retain actual costs instead of dropping or truncating them.

Each numeric term can reference an exact evidence-file hash. Reference presence is not proof that the file measures the complete selected learner, full buffer/actor-version behavior, evaluation/checkpoint overhead, memory and hardware/concurrency basis. The report keeps `runtime_prices_independently_qualified=false` and `execution_authorized=false`. Actual pricing qualification and the A2/resource/entry decision remain required. Old partial-buffer or Q-only profiles cannot establish H_P/H_D/H_N just by assigning their hashes to a term.

## Recorded measurement scopes

Each measurement has a unique identity, covered leaf-scope hashes, exact evidence hash, unit, category and actual/reserved state. The same covered scope cannot appear twice in one resource dimension, including across categories or as both a full reservation and a settled actual. A driver total and its phase/leaf subtotals cannot be summed. Settling a case replaces its reservation; it does not add the actual to the full reservation.

CPU quantities are **process-wall seconds**, summed across workers under the recorded convention. The explicit experimental ceilings are 43,200 control seconds, 14,400 resolver/evaluator seconds and 28,800 probe seconds. Neural CPU work, file/infrastructure verification and development are separately visible and receive no invented ceiling. Classifying scope is an input responsibility; formal oracle/evaluator work belongs in its applicable experimental category. Recorded subsets never claim complete project consumption.

GPU-equipped process elapsed sums are a separate quantity. They are not kernel-active time or physical accelerator-hours. When measured device-allocation intervals and actual physical device IDs exist, the calculator unions overlapping intervals per device. Eight concurrent one-hour processes sharing one device therefore have eight process-hours and one physical allocation-hour. A two-device interval contributes to both physical devices. Missing intervals make physical hours unknown; the tool does not divide job time by an assumed concurrency. The interval union measures allocation, not GPU kernel activity.

## Files and command

Plans use `graygo/nfsp-resource-ledger/1`; quantities serialize as decimal strings and unknown values as `null`. The fixed coefficients and optional selection fields are round-trip checked. Usage files carry the same version and a `measurements` list. Inputs are read-only and an output cannot overwrite them.

```sh
.venv/bin/python tools/nfsp_resources.py \
  --plan docs/verification/nfsp-resource-ledger-current/plan.json \
  --usage docs/verification/nfsp-resource-ledger-current/usage.json
```

The [current recorded example](nfsp-resource-ledger-current/README.md) reconstructs all twenty-one recorded experimental control/profile leaves, retains historical failed/unused experiments and adds only the live selected R4 control reservation. It does not count neural work as table controls, add driver subtotals again, or call the subset total project consumption. P/U/B_U are explicit zero device-hour quantities for the recorded route scope; algorithm-specific device prices remain missing. Thus the CPU reservation fits its recorded ceiling while the full device campaign stays uncosted.

## Verification

Tests exercise the exact selected/unselected formula, failed-comparison retention, pilot-prefix accounting, every missing mandatory price, optional extension/reserve/promotion terms, source references, numerical limits and invalid units. Measurement tests cover the actual recorded control-commitment arithmetic, over-budget retention, overlapping rollups, settled/reserved aliasing, concurrent GPU intervals, multiple devices, unknown physical intervals and read-only CLI output protection. Synthetic price arithmetic is explicitly not measured runtime or trained-game evidence.

The full suite passes **366 tests in 270.438 seconds**. All sixteen ledger tests pass again with the final failed-comparison-retention assertion. The [full log](nfsp-resource-ledger-tests.txt), [targeted log](nfsp-resource-ledger-targeted-tests.txt) and [validation/source record](nfsp-resource-ledger-validation.json) are retained. The current 21-leaf/one-reservation example reproduces numerically without writes. Its device campaign stays uncosted and no execution is authorized.
