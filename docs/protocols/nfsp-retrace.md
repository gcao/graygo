# Conditional Retrace implementation

The [complete original reuse comparison](../verification/nfsp-delayed-r1-neural-seeds12-final/README.md) shows neither one-step configuration passes every original delayed-credit seed. The [recorded reuse choice](../experiments/nfsp-retrace-reuse-choice.json) selects R4 for one conditional target comparison, based on the observed propagation and qualified one-step costs. Its three original seeds and 15M-record caps remain fixed. This does not transfer one-step prices to sequence work or select a full new run.

## Target and replay semantics

One-step remains the default. An explicit Retrace configuration uses lambda 0.95 and at most 32 rounds, as specified in design B2. The target uses current online Q for the legal epsilon-greedy policy and the frozen target network for values. All target calculations stop gradients. The first residual has weight one; subsequent residuals use products starting at the *next* action's coefficient. Logged conditional action probability supplies the denominator directly; episode-mode probability is not multiplied into it.

The implementation checks contiguous valid prefixes, Q eligibility, actual legal actions, true-terminal masks, chronological steps and ordinals, shared successor/source observations, and constant player/episode, frozen opponent, own policy and mode within each sequence. It excludes padding from every model forward. A continuing sequence/replay boundary keeps its final bootstrap; only a true terminal removes it.

Each sampled start produces exactly one projected loss target and one loss-bearing forward. Its retained context is additional inference work, and never increases Q reuse or fresh-record counts. Target refresh still follows fresh Q-eligible records, independently of optimizer reuse. Online/target source inference reuses identical adjacent successor observations; extra successors are inferred only at continuing sequence boundaries. The target forward batch bounds inference memory. Board augmentation uses one geometric transform for each start and its entire context.

## Integration and recovery

The shared NFSP backend implements both targets. Task training and the real-board runtime pass explicit target, length, lambda and target-forward-batch settings. These settings are part of manifests and checkpoints; changing targets on resume is rejected. The existing one-step loss/update remains on its original path.

Cumulative trace diagnostics are serialized with the backend. Counts cover loss starts only, separated by episode mode, supplemental origin, replay age and start target/behavior overlap. Sums retain trace mass, sequence record count, start overlap, decoded opponent snapshot-generation age, replay age and continuation overlap. Context-only averages can be derived using continuation records (sequence records minus starts). Opponent actor IDs contain a policy-type bit; staleness decodes the snapshot generation first. These diagnostics describe opponent staleness; own-action importance correction does not correct for a changing opponent population.

CPU regression fixtures cover independent rational arithmetic, first-coefficient indexing, on-policy telescoping, one-step/lambda-zero reduction, continuing boundaries, true terminals, poisoned padding, projection, start-only gradients and malformed sequence rejection. The off-policy fixture uses the actual solved meaningful-branch game's transitions and payoffs. Actual neural task and augmented-board continuations reproduce learning and RNG state exactly; a short full 64-channel/eight-block 9x9 update checks the production shape.

## Commands and remaining qualification

The task, board-profile and bounded board-diagnostic commands accept an explicit Q target and trace settings. For example, this is a small CPU exercise of the command path and opening/ramp, with a short chain and a diagnostic allocation:

~~~sh
.venv/bin/graygo train-nfsp-task --run-dir runs/retrace-diagnostic \
  --purpose profile --task delayed --learner neural --records 8192 \
  --eval-every 4096 --horizons 5 --equivalent-actions 3 --capacity 8192 \
  --hidden 8 --q-reuse 4 --q-target retrace --target-batch 64 \
  --reservoir-open 4096 --handoff-end 6144 --device cpu --time-cap-hours .05
~~~

A full selected experiment needs the original horizons/codebooks/seeds and fresh source-specific complete pre/post profiling, target-specific prices, independent sequence/optimizer evidence and resource checks. Task command budget validation binds the target and, for Retrace, its length, lambda and forward batch. It rejects a one-step price for a Retrace run.

Existing source bridges, primary cohort plans and one-step prices remain attached to their original frozen packages. The six original bounded Gray Go workers continue on their original source and recipe. No original trajectory, checkpoint, case flag, numerical miss or source identity is rewritten. Full target comparison, controls, complete entry validation and all learning/scientific gates remain required before a production pilot.

Legacy one-step recipe recognition and formal Retrace recipe recognition are separate predicates. A full Retrace configuration cannot enter a v1 one-step plan as a primary case. Recognizing its intended fixed settings supplies no independent sequence audit, source admission, resource allocation or pilot permission.
