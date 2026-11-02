# Gray Go: Game Summary and AI Design

**Status:** Proposed research and implementation design. Revised November 1, 2026. Reviewer-reported experiments are identified separately; this revision did not run or independently audit training.

This document incorporates the game owner's specification and capture-resolution clarification supplied on October 31, 2026. These supersede the provisional rule discussion in the [original training notes](gray-go-ai-training-discussion.md). Neural-network dimensions and training settings below are proposed engineering choices, not measured results. Scoring edge cases inferred from the full-board accounting requirement are explicitly identified.

The recommended first implementation is a deterministic game engine and a small residual CNN with circular padding. Validate a learner on solved games before substantive 9×9 training. A cheap real-trainer PPO matrix preflight selects the route; the PPO cost is a conditional scenario, not the expected campaign price. If that preflight fails, use the profiled NFSP route and its full cost ledger in §6. Add opponent modeling and simultaneous-move search after a reliable baseline. All appended reviews are retained as historical records; §8 records adopted changes and qualifications. Their simulation measurements have not been independently reproduced for this revision.

## 1. Game specification

Gray Go is a two-player Go variant with simultaneous commitments, a toroidal board, and a neutral gray faction. Black and white are the decision-making players; gray stones participate in capture and scoring without an independent policy.

### 1.1 Board and turns

- The default board is 9×9. Larger 13×13 and 19×19 boards are future possibilities.
- The board is toroidal: left connects to right and top connects to bottom. Each intersection has exactly four neighbors.
- Both players secretly select a placement or pass, then reveal together.
- A placement selects an empty intersection that is not forbidden for that player. Pass is always available.
- If the players choose different intersections, place their respective colors. If both choose the same intersection, place one gray stone instead.
- The game ends after **two consecutive rounds in which both players pass**, or when **round 200** has resolved, whichever comes first. Any round containing a placement resets the mutual-pass streak, even if that placement disappears and the board does not change.

### 1.2 Groups, captures, and suicide

Each color forms connected groups of its own color. Empty adjacent intersections are liberties. All three colors can remove the liberties of opposing factions; black and white can capture gray, and gray can capture black or white.

**Capture removal precedes suicide removal.** Resolve removals in this order:

1. Place the round's stones and compute liberties.
2. **Capture phase:** simultaneously remove every zero-liberty group that contains no stone placed this round.
3. Recompute liberties on the resulting board.
4. **Suicide phase:** simultaneously remove every group that still has zero liberties.

The owner's clarification for mutual-death situations governs the suicide phase:

> After removing other captured groups, if both groups are dead, then both are treated as suicidal and removed simultaneously.

Identify the remaining dead groups before removing any of them. Do not remove one such group, recompute, and thereby rescue another group that was also dead at the same suicide check. A surviving collision-created gray group is retained; a suicidal gray stone is removed.

A small mutual-capture example, with every unshown intersection empty:

```text
Before the round:

        c3  c4  c5  c6
  r3     .   x   y   .
  r4     W  B1  W1   B
  r5     .   W   B   .

x and y are empty intersections.
B1 has only liberty x; W1 has only liberty y.
Black chooses y; white chooses x.
```

Both original groups then have zero liberties and contain no new stone, so the capture phase removes both together. The newly placed stones at x and y have other liberties and survive.

A capturing group can be saved even when the opponent takes its last liberty in the same round. The owner confirmed this resolution on October 31, 2026:

```text
        c2  c3  c4  c5
  r2     .   W   B   .
  r3     W   p   W   B
  r4     W   B   B   W
  r5     .   q   W   .

Black plays p; white plays q.
```

After placement, the white stone at r3c4 and black's group containing p both have zero liberties. The white stone contains no new stone, so the capture phase removes it. Black's group then has a liberty at r3c4 and survives the suicide phase.

The supplied suicidal-move definition was a placement not connected to a friendly group that disappears by the end of the round. Under the explicit resolver above, its no-change restriction has a simpler equivalent implementation: **when the stone board is unchanged, forbid every non-pass placement from that round for its respective player**.

Why this is equivalent: placements occur only on initially empty points, so an unchanged final board means every new stone disappeared and no pre-existing stone disappeared. A new stone removed as part of a group containing pre-existing stones would have removed those stones too, contradicting the unchanged board. This also handles a disappearing gray collision without defining whose friendly group it belongs to. Pin this equivalence with resolver tests; never forbid an entire removed group merely because one point was played.

### 1.3 Suicidal-move restrictions

Maintain separate forbidden sets for black and white.

- When a resolved round leaves the stone board unchanged, add each player's non-pass placement to that player's forbidden set.
- This includes one player making a suicidal placement while the other passes, and both players making suicidal placements at the same or different intersections.
- Forbidden points accumulate across consecutive no-change rounds.
- A collision-created gray stone that disappears in a no-change round forbids the selected point for both players.
- Pass is never forbidden.

The combined clearing rule with ko is described below. A round that changes forbidden sets or the mutual-pass streak can leave the stone board unchanged; it is still a different complete game state.

### 1.4 Two-round ko history

Keep the last two completed round records:

```text
(board_before_round, black_move, white_move)
```

After the current round resolves, compare its resulting stone board with the `board_before_round` in the saved records. When a saved board matches, add that record's black placement to black's forbidden set and its white placement to white's forbidden set. These are the moves previously taken *from* the repeated board. Ignore pass when adding restrictions.

The current round resolves normally and is never reverted. Both players see the resulting board and their updated restrictions.

For an explicit update sequence, compare against the two records that preceded the current round, then append the current `(board_before_round, black_move, white_move)` and retain the newest two records. If both prior entries match, this design takes the union of their placement restrictions. These history-bookkeeping conventions make the last-two-round rule precise.

Combine suicide and ko restrictions as follows, following the specification's more specific ko clearing condition:

| Resolved stone board | Existing forbidden sets |
|---|---|
| Unchanged from the start of the round | Retain them; add every current non-pass placement to its player's set, plus any matching-history restrictions. |
| Changed, but equal to a saved history board | Retain them; add matching-history restrictions. |
| Changed to a board absent from the saved history | Clear both sets. |

Board comparisons for this rule compare stone occupancy only. Full state comparisons for search also include forbidden sets, history, the mutual-pass streak, and the round number.

**Consequence to document and test:** suppose black passes while white captures, taking board S0 to S1, then white passes while black retakes, returning to S0. The matching saved record contains white's original capture and black's pass. White's original capture point becomes forbidden; the matching record adds no black restriction. This follows the specified outgoing-move restriction and differs from ordinary Go's immediate-recapture prohibition. Preserve this behavior unless the owner changes the ko rule; do not silently implement ordinary Go ko instead.

### 1.5 Scoring

Use Chinese-style area counting: surviving stones plus connected empty regions. There is no prisoner tracking.

- Each black stone contributes one point to black.
- Each white stone contributes one point to white.
- Each gray stone contributes half a point to each player.
- For each empty connected region, determine the set of stone colors bordering it using toroidal adjacency.

Per-point allocation for an empty region:

| Bordering colors | Black share | White share | Basis |
|---|---:|---:|---|
| Black only | 1 | 0 | Black territory. |
| White only | 0 | 1 | White territory. |
| Gray only | 1/2 | 1/2 | Gray territory is split equally. |
| Black and gray | 3/4 | 1/4 | Half belongs to black, half to gray; gray's half is shared. |
| White and gray | 1/4 | 3/4 | Symmetric to black and gray. |
| Black, white, and gray | 1/2 | 1/2 | Explicitly specified equal division. |
| Black and white | 1/2 | 1/2 | Inferred symmetric convention for neutral territory. |
| No bordering stones | 1/2 | 1/2 | Inferred convention for an entirely empty board. |

The final two rows make scoring total for all positions while respecting the stated full-board score total. They are explicit edge-case interpretations rather than rules quoted from the supplied specification.

Every intersection contributes one point in total, so:

```text
black_score + white_score = N²
```

The total is 81 on a 9×9 board. Use integer quarter-point units internally: four units per intersection. This represents all allocations exactly and avoids floating-point tie errors. Black wins when its total exceeds white's, white wins in the reverse case, and equal totals are draws. No compensation or prisoner bonus is added by this design.

### 1.6 Round limit

The two-round ko mechanism restricts only recent repetition, and longer cycles are possible. The owner chose on October 31, 2026 to keep it and bound the game with a round limit instead:

- A 9×9 game ends when round 200 has resolved, if mutual passing has not already ended it. Round 200 resolves normally.
- The forced end is scored by §1.5 exactly as the board stands. There is no dead-stone removal.
- The limit is part of the rules, not a separate research variant. Limits for 13×13 and 19×19 are not yet set.

The limit creates distinct endgame incentives. A trailing player may prefer continued play, and a surviving final-round invasion can change the allocation of a large empty region with no subsequent reply. The effect depends on the actual region geometry, not just the score before the move. Filling exclusive territory can preserve its current area score while reducing that exposure, but tactical safety and timing still matter. Test concrete final-round positions, including captures and collisions. The numerical examples in the appended review are reported experiments, not consequences that can be derived from the starting score alone.

## 2. Mathematical model

The public pre-round state is:

```text
s = (
  stone_board,
  black_forbidden_points,
  white_forbidden_points,
  last_two_round_records,
  consecutive_mutual_pass_count,
  rounds_played
)
```

Each round has two action sets, `A_B(s)` and `A_W(s)`, and one joint transition:

```text
next_state = resolve(s, black_action, white_action)
```

Both actions must be selected from the same pre-round information. On a 9×9 board, each player has at most 81 placement choices plus pass: 82 actions, or at most 6,724 action pairs before restrictions.

For the proposed training objective, terminal utilities are opposite:

```text
win: +1       draw: 0       loss: -1
```

The specified area scoring is constant-sum. Score difference and terminal win/draw/loss utility both give zero-sum learning formulations. Use win/draw/loss as the primary reward so the policy maximizes expected game outcome rather than expected winning margin. Gray stones can still benefit one player strategically, despite their neutral color.

The board is publicly observed between rounds, while the opponent's current commitment is hidden. This is a simultaneous-move game with imperfect information at commitment, without necessarily having private cards, private objectives, or persistent hidden board state.

A policy maps available information to a distribution over the player's own actions:

```text
pi_B(action | public information)
pi_W(action | public information)
```

With fixed policies and independent private random draws, their conditional joint distribution is the product of those two distributions. Strategic dependence between the policies does not itself create statistical dependence between the draws.

An opponent prediction `q(b)` and a hypothetical move-pair value `Q(s,a,b)` are different objects from the executable policy. Against a fixed prediction, evaluate each candidate action across all possible opposing moves:

```text
expected_value(a) = sum_b q(b) * Q(s, a, b)
```

Do not sample an imagined opponent move and mechanically play its best response as though that move had been observed. That procedure need not be a best response to the actual distribution.

## 3. Neural network

### 3.1 Design goals

The baseline should:

- Respect toroidal adjacency.
- Share parameters between black and white.
- Produce probabilities over the complete selectable action set.
- Estimate expected outcome from the acting player's perspective.
- Remain small enough for many parallel self-play games.
- Use only information available before the current commitments.

The dimensions and optimizer settings below are starting hypotheses, not measured optima.

### 3.2 Inputs

Encode all rule-relevant state from the acting player's perspective: own stones, opposing stones, and gray stones. Use an `N×N×C` spatial tensor and a small vector of public global features.

| Feature | Channels | Notes |
|---|---:|---|
| Current occupancy | 4 | Own, opponent, gray, empty. |
| Current group liberties | 12 | For each faction: one, two, three, or four-plus liberties. Empty points have zero in these channels. |
| Saved pre-round boards | 8 | Four occupancy channels for each of the two history entries, ordered by age. |
| Saved revealed placements | 4 | Own and opponent placement maps for each history entry. Pass has no marked intersection. |
| Current forbidden points | 2 | Own and opponent forbidden sets. |

This gives **30 spatial channels** before broadcasting global features.

Global features are:

- Two history-valid flags, distinguishing an absent entry from an empty board.
- Four pass flags: own and opponent for each saved history entry.
- The consecutive mutual-pass count: zero or one in a nonterminal position; two is terminal.
- Rounds remaining before the round limit, normalized by the limit, plus six threshold flags `remaining <= k` for `k` in `{1, 2, 3, 5, 10, 25}`.
- Board size if training across multiple sizes.

At a nonterminal pre-round state, `remaining = 200 - rounds_played`, so `remaining = 1` means the pending round is round 200. Terminal states are handled without another policy call. Threshold flags are redundant public features that make late-game boundaries explicit; they do not establish that the normalized scalar alone is inadequate. There are 14 scalar globals for fixed 9×9 play: two history-valid flags, four pass flags, one mutual-pass count, one normalized countdown, and six countdown thresholds. Broadcasting these alongside 30 spatial channels gives 44 encoder input channels; variable board size adds another global.

Missing history is zero-filled and identified by its validity flag. Encode every historical board from the same current player's perspective as the present board. No prisoner count or compensation feature is needed for the specified game. The rounds-remaining feature is required: the same board has different values and best moves near the limit.

Broadcast global features into the input tensor and also provide them directly to the pooled output heads.

This represents the complete rule state under the stated history conventions. Two identical stone boards can have different legal actions or different future ko consequences, so the forbidden sets and historical boards are essential inputs rather than optional recent-history hints. A recurrent model is not required merely to remember the game's ko state; longer memory remains an optional tool for modeling an opponent.

Derive the policy mask from the current board and the acting player's forbidden set. Empty, non-forbidden intersections are selectable; pass is always selectable. Do not preemptively mask an otherwise selectable placement merely because it might be suicidal. Whether it disappears, and whether its location becomes forbidden, is determined after joint resolution.

**Legality must not leak the opponent's hidden commitment.** Compute both masks before either current action is sampled or revealed.

### 3.3 Shared encoder

Proposed baseline:

```text
spatial features + broadcast global features
    -> 3×3 convolution, 64 channels, circular padding
    -> GroupNorm + ReLU
    -> 8 residual blocks
    -> shared N×N×64 feature map
```

Each residual block contains:

```text
3×3 circular convolution, 64 channels
GroupNorm + ReLU
3×3 circular convolution, 64 channels
GroupNorm
add block input
ReLU
```

Circular padding connects the board's displayed edges. Group normalization avoids dependence on rollout-batch composition. Start with this convolutional encoder; add attention only if measured weaknesses justify the additional complexity.

### 3.4 Policy head

```text
shared feature map
    -> 1×1 convolution, 32 channels + ReLU
    -> 1×1 convolution, 1 channel
    -> N² placement logits

global average pool(shared features) + global features
    -> MLP, 64 hidden units
    -> 1 pass logit

concatenate -> apply selectable-action mask -> softmax
```

On the target board, this yields 82 action probabilities. Occupied or forbidden placements receive zero probability. Pass remains available in every nonterminal state, including when all placements are masked.

Sample one action from the policy using private randomness. Evaluation should retain learned mixing at temperature 1. Automatically taking the highest-probability move can destroy a useful mixed strategy.

### 3.5 Training critic and search value

Use an opponent-conditioned critic if PPO is selected; the NFSP branch in Phase B2 uses action values and an average-policy network instead:

```text
public shared encoder
    -> global average pooling
    -> concatenate public globals and training-only opponent-bucket features
    -> MLP, 64 hidden units + ReLU
    -> linear scalar + tanh = V_train(s, opponent_bucket)
```

Start with buckets for current self-play, historical age bands, fixed baseline families, and counterstrategy opponents. Freeze the mapping for a rollout batch and record the actual checkpoint identity as well. A bucket averages over its members; it does not identify the opponent's entire strategy or remove all value uncertainty.

The bucket enters only the critic branch after pooling. It must not enter the shared encoder, actor, masks, deployed opponent predictor, or action selection. It also excludes the current sampled actions. This is a proposed variance-reduction aid, consistent with centralized training and decentralized execution; improvement still needs measurement. [Multi-agent actor-critic research](https://arxiv.org/abs/1706.02275)

The output is in `[-1, 1]` and estimates policy-dependent terminal utility for the learner against its training opponents. Record return error and advantage variance by bucket. A fixed self-play bucket is still a policy-evaluation value, not a minimax value.

For Phase E, use a separately identified **public-input search value** `V_search(s)`, trained on outcomes or search targets from a documented frozen strategy pair or population. It may be initialized from the public part of the training critic, but must be calibrated on that continuation distribution. Selecting a self-play bucket alone does not turn a pool-conditioned critic into an equilibrium evaluator. Neither learned head has a full-game minimax guarantee.

Both players act every round, so a trajectory followed from one fixed player's perspective does not alternate value signs between rounds. Changing the perspective from black to white does reverse zero-sum utility.

### 3.6 Optional final-score head

The scoring rule also supplies an exact terminal target:

```text
normalized_final_margin = (own_score - opponent_score) / N²
```

An optional scalar head can learn this target with a separate mean-squared-error loss, initially weighted at `0.1`. Train it from completed games, and evaluate whether it improves the shared representation. Keep the primary actor reward and value head tied to win/draw/loss. Optimizing expected margin is a distinct objective and can prefer different risks.

### 3.7 Symmetry

The specified stride-one circular CNN, spatially shared normalization and pointwise heads are translation-equivariant; average pooling makes pass and scalar outputs translation-invariant. Uniformly broadcast globals preserve that property. Therefore, omit translation augmentation and verify it with a unit test over toroidal shifts of the **entire** state. Require placement probabilities to shift with the board while pass and scalar outputs remain unchanged, within a declared floating-point tolerance.

On the initial empty state, with no asymmetric history or forbidden points, this implies equal probability for every placement point. The pass probability is separate; the policy is not forced to be uniform over all 82 actions. Later states can break spatial symmetry through stones, history, or restrictions. Revisit the equivariance test if absolute coordinates, spatially varying parameters, or strided operations are introduced.

Keep rotations and reflections as data augmentation because this encoder does not enforce them exactly. Transform the current board, both historical boards, historical moves, forbidden sets, masks, and placement labels together; pass remains pass.

For PPO, choose a consistent transformed coordinate system for an entire rollout episode. Do not augment observations during PPO updates while retaining an old action log-probability computed in a different coordinate system. Supervised policy distillation can use paired input/target augmentation.

The supplied scoring and restrictions are symmetric between black and white. Color exchange swaps player-owned stones, forbidden sets, and historical moves while preserving gray, geometric coordinates, the mutual-pass streak, and the round number. Reverse score difference and utility when changing the evaluated player.

## 4. Training process

### Phase A: Establish a trustworthy environment

Before optimizing a neural policy:

1. Implement a deterministic reference resolver with an explicit rule-version identifier. Validate the run manifest, checkpoint restore/rotation, interrupted-save handling, and disk-reserve guard alongside the engine before long training.
2. Test wrapping adjacency, groups, liberties, collisions, captures-before-suicide, and simultaneous removal of the remaining dead groups.
3. Test the equivalence of the no-change restriction and forbidding every current placement, including disappearing collisions. Test accumulation, the outgoing-move ko example in §1.4, the last-two-round history boundary, always-legal pass, and the combined clearing condition.
4. Test that exactly two successive mutual-pass rounds terminate play; placements that disappear must reset the streak. Test that play ends after round 200 resolves and is scored as the board stands, including when that round contains placements.
5. Test every scoring boundary set, fractional allocations, and the invariant `black_units + white_units = 4 * N²`.
6. Check that resolution is independent of container iteration order and the order in which the two commitments arrive.
7. Check translation, rotation, reflection, and color-exchange invariance for the entire state, including scoring and restrictions.
8. Build uniform-random and simple tactical agents.
9. Create small-board, short-horizon problems that can actually be solved by exhaustive enumeration and matrix-game solving.

Do not assume an unrestricted 3×3 or 5×5 game is automatically easy to solve: captures and repetition can still create long games. Explicit horizons make the verification problems finite.

### Phase A2: Select and verify a learner on solved games

Use real trainer implementations in a **staged route-selection process**. Start with tabular controls, then the PPO matrix preflight below. Do not assume that the nominally cheaper PPO route will pass, and do not spend NFSP's full allocation when a verified PPO route already meets the requirements. The conditional NFSP route and stop/go decisions are specified in §6. Use five preregistered task families: an asymmetric 2×2 matrix game with a nonuniform mixed solution, rock-paper-scissors, two bounded Gray Go subgames covering pure and genuinely mixed solutions, and the delayed-terminal-reward family below. Matching pennies is an additional quick unit test, not a substitute for the asymmetric task. Record the actual matrices, roots, horizons, payoff scales, and exact solutions. Verify the exact evaluator against a direct matrix solver and a tabular reference learner before testing neural agents.

A2 validates the algorithm implementation, action masks, terminal handling, opponent-sampling semantics, and deployed policy representation. It does **not** certify one rollout size, optimizer setting, or network size for every task. Matrix tasks may use a small input adapter and compact encoder; bounded Gray Go tasks use the board encoder. Reuse the production loss/update, replay, masking, and sampling code. Document all task adapters and A2-versus-9×9 setting differences, and include a short 9×9-shape training smoke check.

**PPO route preflight:** run the asymmetric 2×2 and rock-paper-scissors tasks with three seeds under two fixed protocols: (a) the existing normalized-entropy taper `0.01 -> 0.001`, and (b) fixed normalized-entropy weight `0.1`. Keep the same rollout, optimizer, clipping, and `0.02` KL stop in both; do not confound the entropy comparison with disabling the stop. Grade latest and declared episode-mixture policies on the original unregularized game. A protocol must pass both tasks, not merely a different task for each setting. These runs use the production trainer's loss/update/masking code, not a new toy implementation.

If neither protocol passes at the declared budget, record that these PPO configurations failed route selection and plan around NFSP thereafter. This does not prove that every PPO configuration fails, and it does not license an unbudgeted entropy sweep. If one passes, carry that protocol into the remaining A2 tests; any subsequent change must be recorded and rechecked. The review's failure rates remain reviewer-reported until reproduced.

A2 uses task-specific settings and reports updates as well as fresh records. The short-task defaults are below; the harder delayed-reward family uses its production-shaped settings and larger allocation in §6:

| Setting | A2 PPO default |
|---|---|
| Rollout batch | 1,024 learner decisions |
| Minibatch / epochs | 128 / 4 |
| Optimizer / clipping | Adam `3e-4` / PPO clip `0.2` |
| Discount / GAE | `1.0` / `0.95`; on a one-step matrix task the advantage is terminal reward minus baseline |
| Entropy protocol | Compare taper `0.01 -> 0.001` with fixed `0.1` on the matrix preflight, then retain the selected passing protocol |
| Opponents | 50% frozen current policies and 50% learning-history checkpoints; no fixed heuristic baseline component |
| Budget per short task family and seed | 2M aggregate records; delayed-reward family: 10M with the production rollout size, separately budgeted |
| Evaluation schedule | Every 102,400 aggregate fresh records, at the next completed update boundary, and at the final checkpoint |

The NFSP-style branch has a **15M-record** cap per short task family and seed. Delayed-reward target comparisons have their own allocations in §6. Inspect its learning curves at 2M, 5M, and 10M, and evaluate exact gaps every 500,000 decisions and at the final checkpoint. Run three seeds for each learner/task family. The final five NFSP evaluations span the end of this longer run. §6 reserves a larger time cap; if profiling shows that the intended decision budget cannot fit, classify the run as budget-limited before using it to select or reject a learner.

For every A2 task, record whether parameters are shared across roles or separate. The stated decision cap is **aggregate over both players**, with each recorded own action counted once. A shared parameter set increments one update counter from all its training records; separate learners each increment only from their own records and split the aggregate decision allocation evenly. Thus 15M means 7.5M per player for a balanced two-learner task. Save per-role counts, update counts, and target-refresh counts; do not compare runs using an unspecified counting convention. The review's 10M toy NFSP result used a different counting/update arrangement and is evidence for adding margin, not a reproduced runtime or a guarantee at 15M.

Run a **tabular NFSP control** on the same tasks, modes, exploration, sampling, and counters. Its Q estimates are tabular, and its average policy averages the frozen greedy target distributions in the reservoir. Keep an executed-action-label control on the pure-solution task to isolate the exploration-label effect. An exact-best-response control can further separate Q-learning delay from averaging delay. If a control is still improving when capped, record inadequate budget rather than declaring a neural implementation bug. If controls pass and neural agents do not, inspect function approximation and optimization. Retain the fixed original-game gap threshold; more data does not justify silently relaxing it.

PPO entropy is an explicit trial, not a promised anti-collapse mechanism. Both the small floor and the higher fixed setting alter the training objective; judge them by support, KL, and unregularized gaps. The review motivates early route selection, not a universal claim about PPO. Neither PPO averaging nor neural NFSP is assumed to converge on every task. [Cycles in adversarial regularized learning](https://arxiv.org/abs/1709.02738)

**Delayed-terminal-reward family:** use solved chains of 50, 100, and 200 rounds, with an initial action selecting a terminal outcome and zero intermediate rewards. Include role-swapped versions and all horizons within one declared family budget. The decisive value checks must not be bypassed by an answer-revealing observation:

- Give each state a fixed opaque one-hot identifier or independently generated random code. Permute identifiers independently of reward sign, branch, depth, and action quality. Expose legality and, if needed, countdown separately, but never a branch/win flag, accumulated terminal return, or outcome-correlated ID range. Use several independent codebooks across seeds.
- Keep the original branch-visible adapter only as an easy generalization diagnostic. It cannot satisfy the delayed-credit requirement. Test the opaque-state adapter with tabular controls and inspect early-state errors separately from policy gaps.
- Retain a forced-action chain as a target-indexing check. The required stress case gives intermediate states 40 equivalent legal actions that lead to the same next state, and uses the real best-response/average-policy mode mix. Measure actual behavior/target overlap rather than assuming that many actions necessarily cut traces.
- Add a small solved branching control where a later action changes the eventual payoff. An uncorrected multi-step return can look excellent on action-equivalent chains because subsequent choices do not matter; that does not establish correctness on Gray Go's meaningful branches.

**Provide coverage for the value test.** Within each balanced horizon/role allocation, use randomized blocks of ten chain episodes: eight ordinary root-policy episodes, one value-only favorable-branch episode, and one value-only unfavorable-branch episode. Generate each supplemental episode by taking the real root transition with the specified first action, then continuing with the normal learner behavior. This supplies the root-action edge as well as the branch states, without adding an outcome flag to observations.

- PPO receives **no actor loss** from a supplemental episode. Exclude its forced root transition from PPO critic fitting too, since that return is not sampled from the actor's root distribution. Use only the normally generated branch suffix for critic learning. Compute actor advantages and normalization from ordinary eligible trajectories only. Shared-encoder value gradients can still change the actor; evaluate the resulting policy normally.
- The Q learner may use the forced root transition and its suffix: these are legitimate off-policy action-value samples. Log the actual forced-action behavior probability and subsequent probabilities. Exclude the entire supplemental episode from the averaging reservoir, even if its suffix uses best-response behavior.
- Mark supplemental records explicitly and count them within the same fresh-record/time budget. Report visits, replay samples, and value errors separately by branch, horizon, root action, and ordinary versus supplemental origin. Policy-gap evaluation always starts at the unforced root. Passing this supported-coverage test does not establish adequate exploration of all branches in 9×9 play.

For NFSP, first compare one-step learning at `R=1` and, when needed, `R=4`, each with the same 15M-record cap per seed. Consider Retrace only after the one-step reuse decision as specified in B2; the uncorrected n-step diagnostic is conditional rather than mandatory. Use identical target refresh, exploration, replay sampling, codebooks, and evaluation states where applicable. The diagnostic is not a deployment candidate merely because it propagates rapidly. Inspect both native online data and declared off-policy replay fixtures, including low target/behavior overlap. With nonzero epsilon, Retrace weights generally shrink rather than becoming exactly zero on every nongreedy action.

Grade **delayed credit separately from averaging**. At each horizon require at most `0.10` absolute early-state value error on both supported branches, including declared root-action values for the Q learner. Compare Q with exact values for its current target policy and PPO's critic with exact values for its current rollout policy. Require a `0.05` root policy-value gap for PPO's deployed actor/mixture and for NFSP's **greedy Q policy without epsilon**. NFSP's average-policy gap does not change the delayed-credit result, but it now has a separate A2 deployment-readiness decision below. The average policy must also pass the short-game averaging gates and remains the policy graded at 9×9 screening and scientific evaluation.

A whole-history reservoir can retain labels collected before long-delayed Q values become useful. That is a different timescale from learning Q itself. Under an ideal repeated-root calculation, random labels for an initial fraction `t` give a gap near `t` when errors cost two utility units; actual state visitation and later labels determine the real effect. Do not turn the delayed-credit family into an implicit requirement that Q be correct in the first 5% of a run.

**Primary A2 averaging protocol before 9×9:** after selecting Q reuse, test **Q-driven pre-opening, a 5M-record reservoir opening, and a 12,800-fresh-Q-record target refresh**. Before a usable average-policy network exists, both players use their own frozen epsilon-greedy Q policy in both episode modes, with the existing exploration schedule and independent draws. Keep the mode tags for label eligibility and log which policy actually produced each action; a mode tag alone does not identify the pre-opening behavior.

The reservoir remains closed before 5M aggregate records, including in best-response mode. At opening, insert only ordinary best-response-mode frozen greedy labels; supplemental value-only episodes remain excluded. Average mode continues using epsilon-greedy Q until the first supervised update, then samples its learned average policy. Q learning runs throughout. A possible distribution shift when the average comes online is part of the pilot to measure, not a reason to infer deployed strength from a Q-only win rate.

Retain **uniform pre-opening as a matched comparison arm**, with the same 5M opening, refresh, reuse, seeds, and environment. It changes only the policy used in average mode before a usable average exists: uniform legal sampling instead of epsilon-greedy Q. A whole-run reservoir opening at zero is an optional reference, not an automatically required precursor to the primary trial. Comparators need not pass the primary readiness gate to be informative.

These changes are motivated by the eighth review's reported Q-only 9×9 and toy results, not independently reproduced findings. The real trainer must still pass the short mixed tasks, bounded Gray Go subgames, and meaningful-branch control. The reviewer did not test all of those with the complete new configuration. Terminal-only matrix tasks cannot validate a target-refresh interval.

Reuse prior results only when behavior, opening, target refresh, reuse, seeds, and other settings match exactly. Test the primary protocol on the short averaging tasks and delayed family, three seeds each. The short-game deployed-policy requirement remains `0.05`; use the following separate long-chain average-policy readiness bands over all horizons/seeds and the final five evaluations:

| Worst deployed average-policy gap on the chain | Decision before a 9×9 learning pilot |
|---|---|
| At most `0.10`, with short averaging and Q/greedy gates passed | Eligible for a limited 9×9 pilot; not a claim of full-game strength. |
| Above `0.10` and at most `0.25` | Averaging-lag warning. Do not advance to 9×9 learning or strength screening; finish the declared protocol comparison/diagnostic work. Profiling inference/update cost is still useful. |
| Above `0.25`, or a short mixed-game gate fails | No-go for that deployment protocol. Use an alternative passing protocol/learner, or stop at the stated scope limit. |

These are preregistered engineering tolerances, separate from the delayed-credit grade. Do not choose a protocol that improves the chain by sacrificing the mixed tasks. Q can pass delayed credit while the deployment route remains blocked. Do not wait until a 20M-record 9×9 screening to discover that distinction.

**Retire the 7M fallback.** The reported 7M tests did not repair the uniform-opponent weakness or the delayed Q target problem, while leaving less time for averaging. That evidence does not support retaining this fallback. There is no 7M trial or `CS7` allocation in the live plan. A failed primary configuration calls for the declared reuse/target diagnostics and a recorded scope decision, not automatic movement of the opening point.

**Marginal-window rule for a 15M-record NFSP protocol, fixed before running:** a candidate may receive one stability extension only if exactly one family and one seed miss the final-five criterion, the final evaluation passes every threshold, and every earlier miss in that five-evaluation window is at most 20% above its threshold. Thus `0.12` is the maximum marginal miss for a `0.10` requirement, and `0.06` for a `0.05` requirement. An excursion such as the review's `0.322` value error is not marginal. Continue all three paired seeds of that affected family for at most **2M additional aggregate records per seed**, with unchanged parameters, opening history, exploration floor, and evaluation intervals. Require the final five evaluations at the extended endpoint to pass; do not stretch the original exploration schedule, choose a lucky intermediate checkpoint, or extend again until success. Report the actual 17M endpoint rather than claiming a 15M pass. Budget this conditional extension before the run; incomplete time-capped extensions remain inconclusive.

This rule distinguishes a small stability miss from a substantial error; it does not assert that more data will work or reintroduce a later-opening fallback. The review's rough label-lateness calculation depends on error rates and visitation, so it is not used as an exact prediction of the required opening point.

Use production replay/rollout paths and cadence with a compact opaque-state encoder. PPO uses its 32,768-record rollout here and has 10M records per seed, about 305 pooled complete updates. Its decision-based evaluation clock remains every 102,400 aggregate records; the final five evaluations therefore do not depend on an impossible 100-update interval. NFSP evaluates every 500,000 records. Five consecutive eligible final evaluations are required. These tests diagnose propagation and target bias; they do not establish full-game strategic ability.

For utility `u(p,q)` to the maximizing player and solved value `v*`, evaluate:

```text
row_gap(p)    = v* - min_q u(p,q)
column_gap(q) = max_p u(p,q) - v*
```

For multiround tasks, optimize over complete opponent strategies. For PPO, evaluate the latest actor and a predeclared episode-level checkpoint mixture. Use a uniform mixture over the scheduled evaluation checkpoints in the latter half of the run; at an interim evaluation use the available checkpoints from that interval. Select each player's checkpoint independently once per episode. For NFSP, the short-game gate evaluates its learned average policy, while the delayed-credit gate evaluates Q and its greedy policy as specified above; report the other policy in each case as a diagnostic. A distilled PPO mixture is evaluated separately if used.

An episode mixture is different from averaging network weights or choosing a fresh checkpoint every round. Distillation must reproduce the mixture's history-dependent behavior as closely as the representation permits and be tested for lost strength. It is compression of a strategy, not a reliable cure for a mixture that already fails the gap threshold.

**Acceptance:** in all three seeds and the final five scheduled evaluations, apply the `0.05` exact gap threshold to the deployed policy on short-game tasks and to the specified actor/greedy-Q policy on the delayed-credit family, together with that family's `0.10` supported-branch value-error requirement. Evaluate the original unregularized game objective. A time-capped run without five eligible evaluations is budget-limited, not a failed convergence theorem. For NFSP, passing the Q/greedy and short-game checks is not enough: the separate average-policy readiness decision must also permit a pilot. A passing route is not automatically selected for the full campaign. Profile the selected eligible implementation at 9×9, including update/replay costs. Profile a second learner only when the staged route selects that fallback or comparison; do not require both full allocations before proceeding. Select using held-out gaps, pilot results, inference cost, and the algorithm-specific full-study resource envelope in §6. A small toy-game gap advantage does not by itself justify a much larger campaign. Record the choice and feasible scope before confirmatory runs.

**Executable fallback:** if PPO fails and the fallback route is selected, assess NFSP over its own full, larger allocation rather than rejecting both at the PPO cap; if only PPO's mixture passes, retain that mixture unless verified distillation also passes. If neither learner passes, freeze 9×9 expansion and compare their masks, targets, gradients, and exact gaps against the tabular controls on the smallest failing task. Allow one documented corrective rerun under the reserve in §6. If that still fails, report a blocked learner-selection milestone and the failing task; do not promise that more compute or distillation will repair it. Keep independent gate tasks for regression testing so tuning does not only fit the five selection families.

### Phase B1: PPO branch for 9×9

If PPO is selected by A2, use it with the network above and treat the following larger-batch settings as a separately validated 9×9 starting configuration. PPO collects experience with a frozen behavior policy, then performs multiple minibatch updates using a clipped policy objective. Its original results do not guarantee convergence to a game-theoretic equilibrium in Gray Go. [PPO paper](https://arxiv.org/abs/1707.06347)

For each rollout batch:

1. Freeze a copy of the current learner.
2. Assign an opponent and learner color at episode start.
3. Build both observations from the same pre-round state.
4. Obtain both distributions before either action is revealed.
5. Sample the actions independently and resolve the round once.
6. Store observations, masks, sampled actions, behavior log-probabilities, training-opponent identity/bucket, critic estimates, rewards, and terminal versus rollout-truncation flags.
7. Compute advantages using the same critic conditioning for each trajectory and update the learner after collecting the batch. Keep bucket metadata outside actor inputs.

In latest-versus-latest games, both perspectives may supply learner trajectories. Against a historical opponent, use the current learner's actions for PPO updates; historical actions are not fresh on-policy learner samples.

Start with sparse terminal rewards. Avoid separate penalties for collisions, passing, or gray stones, and avoid rewarding captures as a substitute for winning. Such shaping can change the desired strategy. Tactical supervision can be evaluated separately without redefining the main reward.

Proposed initial settings:

| Setting | Starting value |
|---|---|
| Optimizer | Adam, learning rate `3e-4` |
| PPO clipping | `0.2` |
| Minibatch size | 512 learner decisions |
| Rollout batch | About 32,768 learner decisions |
| Update epochs | 4 |
| Discount | `1.0` for the undiscounted terminal-outcome objective; every game ends by round 200 |
| GAE parameter | `0.95` |
| Value-loss weight | `0.5`, mean squared error against return targets |
| Entropy protocol | Carry the A2-selected taper or fixed setting into the pilot; record and recheck any change rather than silently reverting to the failed setting |
| Gradient norm limit | `0.5` |
| Update safeguard | Stop further PPO epochs when measured policy KL exceeds about `0.02` |

Define normalized entropy as `H(pi) / log(number_of_selectable_actions)` when at least two actions are available, and zero when only one is available. Tune the coefficient empirically; do not force the final policy to be nearly uniform.

Bootstrap the value at rollout-chunk boundaries when the underlying game continues. Do not bootstrap after the second consecutive mutual-pass round or after round 200: both are true terminal states, scored by §1.5.

### Phase B2: NFSP-style branch with greedy-label averaging

NFSP combines approximate best-response learning and supervised average-policy learning from the agent's own behavior. The replay and averaging reservoir serve different purposes; the reservoir represents accumulated behavior rather than only a recent window. [NFSP paper](https://arxiv.org/abs/1603.01121)

Use a public-input action-value network `Q(s,a)`, a target copy, and a separate average-policy network using §3's encoder and masked head. Neither actor receives opponent identity, and this branch does not use the PPO critic. Greedy-label averaging and the candidate multi-step update below are explicit engineering variants, not reproductions of published NFSP guarantees or results.

For legal actions, use `Q(s,a) = tanh(z(s,a))`, including the pass output. The target copy has the same bounded head. Terminal-only utility with discount one lies in `[-1,1]`; project every computed regression target to that interval before the loss. This prevents out-of-range bootstrap values, but does not eliminate within-range overestimation, missing data, or opponent nonstationarity. Record raw-target excursions and the fraction projected; validate reward/terminal accounting so projection cannot silently hide an engine bug.

| Setting | A2 one-step matrix tasks | A2 multiround/chain tasks and 9×9 |
|---|---|---|
| Best-response episode probability `eta` | 0.1 | 0.1 |
| Best-response exploration | Legal epsilon-greedy; epsilon `0.1 -> 0.01` over first half of budget | Same |
| RL replay / supervised reservoir | 200,000 / 200,000 records | 500,000 / 500,000 records, subject to profiling |
| Learning start | Q starts after 4,096 Q-eligible records; supervised learning starts independently when its reservoir has a minibatch | Same |
| Update counter | One 128-target Q update and one supervised update per 128 new eligible records | One 512-target Q update per 512 fresh Q-eligible records (`R=1`); selected `R=4` comparison uses four such Q updates; supervised cadence remains one batch per 512 ordinary records |
| Q minibatch | 128 loss-bearing starts | 512 loss-bearing starts; one transition each for one-step, up to 32 context transitions each for Retrace |
| Supervised minibatch | 128 reservoir records | 512 reservoir records |
| Optimizers | Adam `3e-4` separately | Same starting rates |
| Hard target refresh | Irrelevant to terminal-only one-step targets; no claim about refresh follows from those tasks | Every 12,800 fresh Q-eligible records per parameter set: 25 Q updates at `R=1`, 100 at `R=4` in steady state |
| Q target | Terminal reward for one-step games | One-step baseline versus Retrace candidate (`lambda = 0.95`, at most 32 rounds), selected by the delayed-credit comparison |
| Discount | 1.0 | 1.0, with no bootstrap at either true game ending |
| Average-policy loss | Masked cross-entropy against frozen greedy labels | Same; no labels from value-only coverage episodes |

Define Q replay ratio `R` as the number of **loss-bearing start targets** optimized divided by fresh Q-eligible records. Context states used only to build a sequence target are extra compute, not extra supervised starts. The former 64-per-512 setting was `R=0.125`; start the one-step comparison at `R=1`. If it does not pass the supported delayed-credit gate, test `R=4` with the **same 15M-record cap per seed**, rather than the previous 5M screen. Hold target refresh fixed in fresh-record units. Otherwise a reuse change would also accelerate target refresh and confound the comparison. Use microbatches as needed for 512 sequence starts; count their combined optimizer work correctly. Do not assume that increasing Q reuse fourfold has the same total runtime multiplier as the full actor/critic pipeline.

Each player independently selects best-response or average-policy mode at episode start. Before the first usable average-policy update, the primary protocol acts epsilon-greedily from Q in both modes; afterward, best-response mode remains epsilon-greedy and average mode samples its learned policy. The matched uniform comparison overrides only pre-opening average-mode action selection. **The action executed and the averaging target are separate records:**

1. Before a best-response-mode action, compute a target distribution uniform over the legal maximizing actions of that frozen behavior-Q snapshot. Freeze this label with the state and version.
2. Once the selected reservoir-opening threshold is reached, insert one such greedy target for every ordinary best-response-mode state visit, including visits where epsilon exploration chooses a different executed action. Train the average-policy network by cross-entropy against these frozen soft labels. Do not relabel historical samples using current Q.
3. In RL replay, retain the action actually executed, its actual behavior probability, reward, and transition. Never replace the replay action or its probability with the greedy label.
4. Average-policy-mode visits contribute RL records but no averaging-reservoir labels. Value-only coverage episodes never enter the reservoir. Standard deployment samples the average policy without external epsilon or anticipatory mixing; the delayed-credit diagnostic separately grades the greedy Q policy.

This removes deliberate epsilon noise from the supervised action labels without discarding exploration-selected visits or weighting epochs by `1-epsilon`. Ties remain randomized. The visited-state distribution still comes from exploratory behavior, so this is not an exact historical-average guarantee for the unexecuted greedy policies; approximation and early poor greedy labels remain. Verify the modified averaging rule with tabular controls and the short-game deployed-policy gap gates.

**Primary reservoir opening:** 5M aggregate records with Q-driven pre-opening. Use the matched uniform arm or zero-opening reference only when selected as comparisons. Test the primary protocol and the three-checkpoint 9×9 opening window in §6.3; the absolute record count alone does not establish readiness. Do not reset populated history or shift opening to 7M after a miss. Protocol validation does not supply a convergence guarantee.

For a repeated state with a fixed unique best action, uniform visits through the declared epsilon schedule yield mean epsilon `0.0325`. Copying executed epsilon-greedy actions would put `0.0325 * (A-1)/A` mass on inferior actions; when all alternatives lose two utility units, the resulting gap is about `0.0642` at `A=82`. This arithmetic is a control calculation, not a universal per-state floor or measured deployed error rate. Stretching the schedule with each larger budget preserves that control's average; extending only its low-epsilon tail would not. Add a pure-solution control with 82 actions and check frozen greedy-label targets directly.

Store compact rule states in host memory and regenerate neural features on sampling. During target comparison, replay sequences must remain within one player's trajectory, episode, and frozen opponent version. Record the behavior probability of the action under the actual mode and policy snapshot used; do not multiply it by the episode-mode probability again. Retain terminal and rollout-truncation flags separately.

**Corrected multi-step candidate:** test a truncated Retrace-style correction against its one-step counterpart; do not assume that adding sequences helps. For a sampled sequence, let `mu` be its logged behavior distribution, `pi` the current legal epsilon-greedy best-response target policy, and `Q_bar` the frozen target network. Stop gradients through the target calculation:

```text
c_i     = lambda * min(1, pi(a_i|s_i) / mu(a_i|s_i))
delta_i = r_i + gamma * sum_a pi(a|s_{i+1}) Q_bar(s_{i+1},a)
              - Q_bar(s_i,a_i)
y_raw   = Q_bar(s_t,a_t)
          + sum_{k=0}^{m-1} gamma^k * product_{j=1}^{k} c_{t+j} * delta_{t+k}
y_loss  = clip(y_raw, -1, 1)
```

The empty product is one. `m` ends at the earlier of 32 rounds or a true terminal; terminal successor value is zero. At a nonterminal sequence boundary, retain the final one-step bootstrap. Mask illegal actions throughout. The current target policy is derived from online Q, while values in this target use `Q_bar`. Verify indexing, one-step reduction, and on-policy behavior against tabular calculations. [Retrace paper](https://arxiv.org/abs/1606.02647)

Use the same legal target policy and frozen target network for every comparison. The one-step baseline is the formula above with `m=1`; an uncorrected n-step diagnostic sums observed rewards and bootstraps at the sequence endpoint without importance correction. Keep target refresh at 12,800 fresh Q-eligible records per parameter set in all target/reuse comparisons, so changing the number of optimizer steps cannot masquerade as a trace benefit. Report both fresh-record and wall-time costs, including the extra target evaluations.

For each sampled sequence, measure effective trace mass `1 + sum_{k>=1} product_{j=1}^{k} c_{t+j}`, separated by mode, replay age, and behavior/target overlap. Equivalent-action chains do not guarantee low overlap; include controlled fixtures and the solved meaningful-branch check. Report early/late-state error and target-refresh counts as well. The target-refresh clock remains 12,800 fresh Q-eligible records per parameter set, including supported-coverage records, independently of replay reuse. Hold this interval constant across the `R=1`/`R=4` and one-step/Retrace comparisons. Apply it to the delayed family, bounded multiround controls, and 9×9; rerun affected checks rather than transferring a terminal-only matrix result. Neither this arithmetic nor a trace length of 32 proves adequate propagation. Neural approximation, finite replay, and changing opponents are outside a simple tabular fixed-environment guarantee; the delayed-reward gate and 9×9 measurements decide adequacy.

**Selection order:** resolve Q reuse with **one-step only** first. If `R=1` passes all delayed-credit and branching checks, keep it unless a separately justified cost experiment is selected. Otherwise complete the paired `R=4` test to its full 15M-record cap, subject to a time ceiling set by profiling. Use the same conditions, seeds, target-refresh clock, and fixed value-error criteria. A 5M checkpoint is progress information, not an adequate substitute for this longer comparison.

If `R=1` passes the Q/greedy tests but its average policy fails the separate readiness bands, keep the Q-pass result. A paired `R=4` run may still be selected to test whether earlier Q learning reduces averaging lag; do not relabel this as failed delayed credit.

If a one-step configuration passes, skip Retrace and the uncorrected diagnostic. If neither one-step configuration passes, select the reuse level for one conditional Retrace test from the observed progress and measured cost, record that choice, and give it the same 15M-record per-seed cap. There is no automatic grid over both targets and both reuse levels. An uncorrected diagnostic may help identify trace/target behavior, but cannot become a deployment shortcut because it wins an action-equivalent-chain race. A selected configuration must pass Q/greedy and meaningful-branch checks, then undergo the averaging-readiness decision. If the profiled work does not fit the resource ceiling, report a budget-limited route rather than silently cutting off the promising configuration or assuming the simple one must suffice.

Use the primary Q-driven episode-mode procedure above, not the PPO fixed-baseline scheduler. Opponent versions remain fixed within an episode. The own-action importance correction does not correct for changing opponent populations across replay: retain opponent-version metadata and assess staleness. If the long-horizon control fails, use the corrective reserve to investigate target cadence, trace truncation, and replay coverage before diagnosing Gray Go strategy or adding reward shaping.

### Phase C: Opponent population and evaluation

For the PPO branch, use a learning-opponent pool and make fixed baselines a short warm-up aid rather than a permanent part of the objective:

| Learner decisions in the declared 9×9 run | Current-policy self-play | Historical learning checkpoints | Fixed heuristic baselines |
|---|---:|---:|---:|
| First 1 million | 40% | 40% | 20% |
| 1 to 5 million | Increase linearly to 50% | Increase linearly to 50% | Decrease linearly to 0% |
| After 5 million | 50% | 50% | 0% |

A2 uses the final 50/50/0 schedule from the start. A smaller nonzero baseline share would reduce, but not remove, the incentive to optimize against that imposed distribution. Even a zero-anchor history pool does not guarantee equilibrium; retain exact gate checks and independent attacks. Archive strong counterstrategies as learning opponents under a versioned sampling plan rather than assigning them an undocumented permanent anchor weight.

Keep opponent parameters fixed throughout an episode. Preserve older checkpoints that expose weaknesses, and keep fixed heuristic agents in **evaluation** throughout training. For NFSP, use Phase B2's native average-policy/best-response self-play; external baselines are evaluation opponents, not an extra training component. Its `eta` is an internal learning mechanism and is not the same as a fixed-baseline share.

The PPO schedule is a population-training heuristic. More formal policy-space response-oracle methods train responses to mixtures and analyze the resulting empirical game; they motivate this direction but do not make the simple mixture equivalent to PSRO. [PSRO paper](https://arxiv.org/abs/1711.00832)

Keep the **training opponent pool**, the **candidate deployment mixture**, and the **protected evaluation pool** as separately versioned objects. Playing against a mixture does not itself average the learner's strategy. Evaluate the 9×9-trained policies using the deployment mechanism validated in A2: a single actor, episode mixture, or learned average policy. The separately trained gate agents are not the 9×9 deployment weights.

Train and report a direct 9×9 baseline. A 5×5-to-9×9 curriculum is deferred unless direct training fails a learning milestone: smaller tori change tactics, and an additional curriculum experiment must answer that specific failure.

### Phase D: Optional opponent prediction

Add this only after the baseline and evaluation pool are stable.

An opponent-prediction head outputs a masked distribution over the opponent's 82 choices from pre-round public information. Train it with negative log-likelihood of the subsequently revealed action. It must never receive that action as an input for the same prediction.

Run two distinct experiments:

1. An auxiliary prediction head sharing the encoder, while the actor retains its original inputs.
2. An adaptive actor that also receives the predicted distribution or a compact summary, and is retrained using those features.

If adaptation requires longer observations of an individual opponent, add memory over previous public rounds. Reset that memory at the declared adaptation boundary; do not leak hidden opponent IDs or training-only policy metadata into deployment inputs.

Measure prediction log loss, calibration, and playing strength. A low top-k coverage rate can reflect a diffuse opponent policy rather than poor prediction.

For explanations, display top-k predicted moves and:

```text
else_probability = 1 - sum(probability of displayed moves)
```

**Explicit design decision:** drop the earlier requirement for a special positive-probability `else` branch and drop the objective of minimizing how often it occurs. The neural policy and opponent predictor already output complete, normalized distributions over all selectable actions. Zero probability on some legal actions is mathematically allowed; exploration is handled separately. The display's `else` is only the summed probability of unshown actions and can be zero. Conditional responses to an opponent's unrevealed current move are analysis, not executable within-round choices. This changes the proposed AI representation, not the game rules.

The NFSP learner in Phase B2 is distinct from an opponent-prediction head. It is a budgeted alternative selected in A2, not a late add-on to fix a failed predictor.

### Phase E: Optional simultaneous-move search

First build a shallow matrix-game baseline rather than immediately engineering a large search tree.

At a public state, evaluate legal action pairs using the deterministic resolver and the calibrated public-input `V_search` from §3.5:

```text
Q[a,b] = terminal utility, if the successor is terminal
         V_search(successor from the same player's perspective), otherwise
```

For a zero-sum objective, compute distributions approximately solving:

```text
max_p min_q  pᵀ Q q
```

Here `p` and `q` range over the respective action simplexes. The full root matrix has at most 6,724 entries; batch successor evaluations and measure their cost. Use exactly scored terminal successors when available, including round-200 positions. Nonterminal leaf values depend on their training continuation policies; the local matrix solve cannot remove that dependence or certify a full-game equilibrium.

Compare this with no-search play under an equal time budget. If it helps, extend to deeper simultaneous-move search using a published algorithm with appropriate exploration and value averaging. Simply inserting a no-regret selector at every node is not by itself a convergence guarantee. [Analysis of simultaneous-move MCTS](https://arxiv.org/abs/1804.09045)

For larger searches, use policy priors to prioritize actions, add tactical and exploratory candidates, and expand coverage with budget. Avoid permanently excluding every low-prior move. Check restricted-search policies against counteractions outside the current candidate set.

Distill improved search distributions into the policy with supervised cross-entropy. Keep this buffer and objective separate from PPO experience: search-selected actions are not samples from the raw policy. Label value-training targets by their origin, and evaluate whether the critic is learning rollout-policy outcomes or search-guided outcomes.

## 5. Evaluation and acceptance criteria

A 50% win rate against an identical self-play copy says little about absolute strength. Use a protected opponent pool, a cross-play matrix, and the exact mixing gate. Evaluate the deployed object, including its private episode-level policy sampling when it is a mixture.

### 5.1 Numerical promotion gate

From the initial empty state, the finite game is symmetric under color exchange and zero-sum, so its equilibrium value is zero. This does not hold for arbitrary asymmetric midgame roots. If evaluating openings, pair them with their color exchanges.

For candidate `i` and fixed opponent `j`, utility is `u_ij = P(win) - P(loss)`. A truly negative expectation against an opponent establishes exploitability from the symmetric start. A negative sample average alone does not; report the worst tested matchup and uncertainty, limited to the tested attack set.

Freeze opponents, evaluation weights, opening generation, confidence procedures, and thresholds before evaluating a candidate. Proposed promotion criteria, in utility units on `[-1,1]`:

1. The selected learning algorithm and deployed representation passed A2 using separately trained gate agents. A2 validates that implementation under its own stated settings, not the 9×9 batch size or an untrained 9×9 checkpoint on unrelated matrix tasks. The 9×9 configuration also passes its own smoke checks.
2. The lower confidence bound for average candidate-minus-incumbent improvement under fixed evaluation weights is greater than zero.
3. For each protected opponent, the lower confidence bound on candidate-minus-incumbent improvement is at least `-0.15`.
4. The worst matchup's lower confidence bound is at least `-0.10`.

The revised `0.15` per-opponent regression tolerance explicitly allows a larger matchup tradeoff than the former `0.05`; it is not statistically equivalent to the old rule. In score-rate units `P(win) + 0.5 P(draw)`, those utility tolerances are 7.5 and 2.5 percentage points respectively. The separate `-0.10` absolute floor remains. Neither rule certifies low full-game exploitability, and widening a margin after seeing a candidate fail is prohibited.

**Calibrate the whole gate before use.** Use pilot outcomes from separately designated agents to estimate draws, paired variance, and dependence across openings. Simulate the exact confidence procedure and all four promotion requirements jointly, including the aggregate improvement test and the absolute floor. Cover zero improvement, a uniform `+0.03` improvement, nontransitive gains/losses, a regression at and beyond `-0.15` in one matchup, and violation of the absolute floor. The review's illustrative pass rates are not transferable guarantees.

Start with 500 and 2,000 color-balanced games per candidate/opponent and incumbent/opponent pairing. If pre-candidate calibration shows insufficient power, budget a final 8,000-game look before collecting candidate data. Include every planned look and protected comparison in simultaneous 95% confidence coverage, and account for matched openings and colors. Target at least 80% power for the preregistered practically useful improvement scenario while keeping false promotion at most 5% when a criterion is violated. If no allowed sample plan meets those goals, report that the gate is not yet calibrated; revise the evaluation budget or substantively justify new tolerances before testing a candidate. Do not turn inconclusive evidence into a pass.

Cap the initial protected set at ten versioned opponents so the sample plan is finite. Broader exploratory cross-play remains useful, but adding an adaptive counterstrategy creates a subsequent preregistered evaluation round. Confirmation uses three independent training seeds; report seed-level results and do not pretend thousands of games eliminate training-seed uncertainty.

Before there is an incumbent, apply §6's learning milestones. Failure to promote does not require discarding a useful specialist from the archive. Numerical tolerances and power targets here are proposed design choices, not measured acceptance rates.

### 5.2 Measurements

Evaluate against random and tactical baselines, historical checkpoints, independently trained policies, and separately trained approximate best responses. Search agents enter once the search prototype exists. A learned best response can miss weaknesses; its result is an attack measurement, not exact full-game exploitability.

| Measurement | Purpose |
|---|---|
| Cross-play matrix and worst matchup with confidence intervals | Detect progress, forgetting, nontransitivity, and exposed weaknesses. |
| Latest/mixture/greedy-label-average exact value gaps on solved tasks | Verify the strategy representation that will be deployed. |
| NFSP average-policy and exploration-free best-response strength, actual label counts, and reservoir age | Separate delayed Q learning from averaging lag; episode probability is not a guaranteed record fraction. |
| Q target excursions/projection rate, loss-bearing replay ratio, and branch coverage | Detect out-of-range targets and data starvation without confusing sequence context with optimized targets. |
| Illegal-action rate and policy entropy | Detect masking defects and policy collapse. |
| Critic error and advantage variance by opponent bucket | Test whether critic conditioning helps. |
| Public search-value calibration | Check continuation estimates separately from the training critic. |
| Collision rate, gray lifetime, and final gray occupancy | Measure whether gray survives or matters transiently. |
| Fraction of scored empty points using gray boundary allocations | Measure how often fractional territory rules affect trained play. |
| Round-200 termination rate and final-round margin changes | Detect dependence on the horizon and endgame invasions. |
| Episode-length distribution, early mutual-pass termination, and draw rate by seed/mode before and after opening | Detect short-game self-play behavior; do not turn game length into an automatic failure or reward penalty. |
| Ko encounters and forbidden-point counts | Characterize the specified repetition mechanism. |
| Games/second, decisions/second, inference latency, and device-hours | Compare methods under measured compute budgets. |

### 5.3 Gray-rule comparison with an exposure control

The collision-removes-both variant leaves the collision point empty, while keeping the 200-round limit and all other rules. No-change rounds still forbid both placements. Gray stones already present in a starting position retain their ordinary capture and scoring behavior in this variant. This is a research variant, not a change to §1.

Use a **2×2 design**: transition rules `{base, collision-removes-both}` crossed with training starts `{empty only, matched gray exposure}`. The original two empty-start arms remain a descriptive comparison. They cannot alone establish gray-specific skill: the variant-trained actor has never encountered a gray input.

For the exposure-controlled pair, both arms use the **same versioned starting-position distribution**: 50% empty starts and 50% gray-containing starts, paired by seed. Build the latter independently of candidate policies, with varied locations and small connected/singleton groups totaling one to six gray stones; include balanced black/white configurations as well. Reject boards with zero-liberty groups. Apply color exchange and toroidal symmetries. Reset ko history, forbidden sets, pass streak, and rounds played to zero for these synthetic starts, giving both arms the same 200-round allowance. These are explicitly curriculum initial states, not claims that the variant can reach them from empty play.

Split training and held-out starting-position families before training. Both arms must receive equal starting-state samples and matched learner-decision exposure. Profile how often they actually see gray groups, captures, liberties, and territory allocations; merely adding gray inputs does not guarantee equivalent coverage of later states. Predeclare a broader exposure-distribution revision only if pilot coverage fails, before the confirmatory comparison.

Evaluate every selected policy under both rule sets from official empty starts, plus the shared held-out gray starts as a diagnostic. Retain original and controlled comparisons side by side, with three matched training seeds per arm. If the original effect disappears after shared exposure, report that it is consistent with unfamiliarity or curriculum effects, not proof that all strategic value disappeared. The control reduces an important confound; it cannot make the two transition systems identical.

Report gray lifetime, terminal occupancy, capture effects, and fractional scoring use. Temporary gray stones can matter even if none survive to scoring. The full four-arm allocation and evaluation caps are explicit in §6.

### 5.4 Equilibrium-consistent collision probes and decision rules

Freeze learners, checkpoint selection, seed pairs, endpoints, and thresholds before confirmatory evaluation. Separate a performance effect from behavior on solved tactics.

**Performance endpoint, exposure-controlled arms:** let `P_G` and `P_E` be the base-rule and variant-rule policies trained on the shared gray-exposure curriculum. Let `u_G` be `P_G`'s color-balanced utility against `P_E` under base rules from empty starts, and `u_E` the same policy pairing under variant rules. Require simultaneous 95% lower bounds above `+0.05` for both `u_G` and the interaction `u_G-u_E`, with positive point estimates for both in all three paired training seeds. Compute the same quantities for the empty-start arms as secondary endpoints. Passing supports a rule-dependent performance effect under the exposure control; neither endpoint alone proves an internal intention to create gray. A weaker or inconclusive controlled effect limits the conclusion accordingly.

**Behavioral endpoint without an oracle opponent:** construct 32 independent families, each containing a pair of completely specified public Gray Go subgame roots. Solve each subgame as a zero-sum game, with its own value and complete-strategy best-response evaluator. Evaluate the deployed learned policy for both roles; do not demand a best response to an arbitrary opponent distribution unavailable to the actor.

At a root, define placement-collision probability by the independently selected action distributions:

```text
P_collision = sum over intersections a of pi_B(a) * pi_W(a)
```

Exclude pass. For an episode mixture, independently sample the role's policy at probe-episode start, or compute the equivalent initial marginal; do not reveal its identity to the other actor. If evaluating after a prefix, use the actual history-conditioned distributions rather than resetting memory or averaging checkpoints afresh.

A single selected Nash equilibrium can be misleading when other equilibria have different collision rates. Before inspecting candidate outputs, compute certified bounds on collision probability over policy pairs whose individual value gaps are at most `0.05` at each probe root. Use the gap formulas in A2. Admit a pair only if these admissible collision ranges are disjoint by at least `0.10`, with a fixed preferred direction. Small one-round games can be checked by optimization over the exact equilibrium/near-equilibrium strategy sets; bounded multiround probes need the corresponding complete-strategy constraints. If range bounds cannot be certified within the probe budget, leave the family ineligible. Balance families where the solved change increases and decreases collision, and avoid counting symmetry copies as independent examples.

A family passes when both roles' exact value gaps are at most `0.05` at both roots and the learned collision probability shifts in the certified direction by at least `0.05`. Require at least 24 of 32 families to pass for each of the three trained seeds. This is a descriptive finite-benchmark acceptance criterion: do not claim a binomial population confidence bound from hand-selected families. With the stronger near-equilibrium range qualification, the direction check should follow from the gap tests; it is a readable behavioral diagnostic and a consistency check, not independent evidence of intent.

An optional adaptive-opponent experiment must be separate and deliver opponent evidence through public history or an explicitly designed observable context. It must not grade the unconditioned equilibrium actor against a hidden oracle distribution.

Passing the controlled performance test and the solved-probe benchmark supports the limited claim of gray-rule specialization and equilibrium-consistent collision behavior on these tests. If 32 certifiable families cannot be constructed, report incomplete coverage. Failure or insufficient precision means not demonstrated at this budget, not proof that gray lacks strategic value.

## 6. Evidence-selected route, profiled budgets, and total cost

These are proposed work ceilings and scope decisions, not measured runtimes. No training or accelerator spending has been started by this document revision. Reviewer toy results motivate the order of work but do not select the real trainer's route. In particular, **do not use the cheaper PPO total as the expected campaign budget before its real-trainer matrix preflight passes**.

**Reported target machine: `home-ai`.** The seventh review identifies an AMD Ryzen 9 5900X with 12 CPU cores, an NVIDIA RTX 3080 Ti with 12 GB of device memory, and 31 GiB of host RAM. It reports approximately 123 GB of free disk at review time. These are supplied environment details, not measurements taken in this revision. Before Phase A profiling, record the live device identity, usable memory, driver/runtime versions, worker count, filesystem, and free bytes in the run manifest. Plan microbatches for the actual 12 GB GPU; preserve the logical loss batch and replay ratio when accumulating gradients. The eighth review supersedes its earlier 35–40-hour estimate with an amortized estimate near ten device-hours per 80M records, based on concurrent Q-only jobs. Neither estimate is a measured complete-NFSP runtime. Profile `H_N` with the average-policy network and selected target/reuse, and distinguish per-job elapsed time, job concurrency, and physical accelerator-hours.

**Counters:** count each new own-side record once, including supplemental value-only records. Report ordinary actor/Q eligibility, actual best-response visits, inserted labels, per-role counts, and replay starts separately. Q reuse counts loss-bearing starts, not sequence context. Shared parameter sets use shared counters; independent role learners use independent counters. Target refresh remains every 12,800 fresh Q-eligible records regardless of replay reuse or worker count.

### 6.1 Order of work and route decision

1. Run CPU resolver/exact-evaluator, masking, bound, replay-indexing, and tabular checks. Catch defects before either neural campaign.
2. Run the PPO matrix preflight with the two declared entropy protocols. If neither passes both tasks, record that outcome and retire the PPO campaign-cost scenario as the planning basis. Do not run the expensive PPO chain merely to exhaust its allocation.
3. If PPO passes, complete its remaining A2 tasks with the selected protocol. A later failure can still choose the NFSP route. A passing PPO route can proceed to its limited pilot without automatically funding NFSP.
4. If NFSP is selected, test the Q-driven/5M primary protocol at the 12,800-record multiround refresh. Profile and budget one-step `R=1`/`R=4` before their full 15M comparisons, then consider Retrace only if required. Complete all applicable averaging, bounded-subgame, and branching checks before 9×9 learning. Uniform pre-opening is a diagnostic comparator, not the primary fallback route.
5. **Before the A2-exit scope decision**, profile 9×9 inference, Q/target updates at the selected reuse, supervised averaging, and replay cost. Fill the NFSP per-seed ceiling and full-study ledger below. A lower-reuse or old 96-hour profile cannot price an `R=4` campaign. Missing cost data is a reason to do profiling or stop, not to substitute the PPO total.
6. Record the selected algorithm, target, reuse, averaging protocol, expected record throughput, resource ceilings, and excluded work. A2 completion selects at most a limited pilot initially, not automatic confirmatory training or a four-arm study.

### 6.2 A2 work allocations

| Component | Fresh-record ceiling | Device-hour ceiling |
|---|---|---:|
| PPO matrix preflight | Two tasks × two entropy protocols × three seeds × 2M = 24M | 24 |
| Remaining PPO short tasks | Two bounded families × selected protocol × three seeds × 2M = 12M | 12 |
| PPO supported delayed family | Three seeds × 10M = 30M | 36 |
| Conditional NFSP primary short tasks, Q-driven/5M | Four families × three seeds × 15M = 180M | 96 |
| NFSP one-step `R=1` delayed family | Three seeds × 15M = 45M | `3*C1` |
| Conditional one-step `R=4` delayed family | Same three seeds × 15M = 45M | `3*C4`; no 5M substitute |
| Conditional Retrace at selected reuse | Three seeds × 15M = 45M | `3*CT` |
| Optional zero-opening averaging reference | Declared short-task/delayed-family comparison, at most three seeds × 15M per selected family | `B_0`, profiled and entered before selection |
| Conditional marginal stability extension | One affected family per candidate protocol, all three seeds × at most 2M additional records | `3*CE` for that selected extension, entered before the run |
| Matched uniform pre-opening comparison | Declared A2 subset and paired 9×9 comparison using the same opening/refresh/reuse | Profiled `B_U`, including work not already covered by the primary runs |
| Optional uncorrected diagnostic | One seed, at most 5M | At most 12, only when a question justifies it |
| One corrective reserve | One documented failing-family correction, at most 45M | At most 24, separately visible |

`C1`, `C4`, and `CT` are per-seed ceilings for complete 15M-record delayed-family runs of the primary configuration, entered from matched profiles **before** launching the comparison. `CE` covers a selected 2M-record stability extension; `B_0` and `B_U` cover the explicitly selected reference/comparison scope. Include evaluation and replay/target computation, and a stated runtime margin. If no feasible ceiling can be entered, the comparison is not yet funded by this plan; do not pretend that an old short screen answers it.

First screen the primary configuration on one seed of the asymmetric matrix, rock-paper-scissors, and the chain, then complete its required three-seed checks if viable. Do not expand a reference arm into a full campaign by default. Apply the marginal-window extension only as declared; it does not change the opening point or waive missing bounded-subgame/branching evidence.

CPU ceilings remain four elapsed hours for resolver/evaluator work and twelve for tabular controls. These are distinct from accelerator hours. A family allocation covers its predeclared horizon/role/coverage mixture, not a fresh cap for every subcondition.

### 6.3 9×9 pilot and confirmation milestones

| Route/stage | Aggregate record milestone | Resource handling | Evaluation |
|---|---:|---|---|
| Verified PPO smoke | 1M | Up to 4 device-hours | Defects and positive lower 95% utility bound versus random. |
| PPO screening | 5M cumulative | Up to 16 device-hours cumulative | Lower 95% utility bound versus tactical baseline at least `+0.10`. |
| PPO confirmation | 20M per seed | Up to 24 device-hours per seed | Three seeds and calibrated deployed-policy checks. |
| Ready NFSP limited pilot | Up to 6M records; save Q at 4.5M, 4.75M, and 5M before opening | At most 16 device-hours initially, or a preselected profiled cap | A2 readiness and the fixed three-checkpoint opening decision; report Q and average policy separately. |
| NFSP screening | Proposed 20M cumulative | Cap set from the selected target/reuse profile | Grade the deployed average policy against random/tactical thresholds. |
| NFSP confirmation | Proposed 80M per seed | Profiled per-seed ceiling `H_N`, recorded before the A2-exit scope decision | Three seeds and matched training exposure. |

The primary pilot seeks one million post-opening records, through 6M total. Its initial time ceiling remains 16 device-hours unless another profiled ceiling is recorded before selection. If it cannot complete the required evaluations or post-opening observation within that cap, report incomplete evidence; do not infer average-policy strength from the Q-only pre-opening phase.

**Fixed three-checkpoint 9×9 opening decision:** archive frozen Q weights at **4.5M, 4.75M, and 5M** fresh records. These are the last three preregistered checkpoints before the 5M opening, 250k apart; no future checkpoint is used to justify opening earlier. At each, evaluate greedy Q without external epsilon against the same uniform-random and tactical opponents, with 500 color-balanced games per opponent and a preplanned 2,000-game look if needed. Use 95% lower confidence bounds with simultaneous coverage across all three snapshots, both opponents, and both planned sample sizes.

A snapshot clears only if its lower utility bound is above zero against random **and** at least zero against the tactical agent. Open the reservoir only if **at least two of the three snapshots clear both checks**. Do not combine a random-opponent pass at one snapshot with a tactical pass at another to manufacture a clearing read. Record every result; the rule uses a fixed window, not the best three snapshots found afterward. Save these evaluation models in the permanent network archive; they do not require three full replay snapshots.

Pause label insertion at the 5M boundary until the decision is complete. If it passes, continue with the current 5M Q state, not a retrospectively selected older winner, and begin ordinary best-response label collection. If it fails or remains inconclusive within budget, preserve the closed-reservoir state and record no-go for that configuration. Do not roll the window forward repeatedly until it passes or move opening to 7M. A changed configuration needs a new declared comparison, applicable A2 rechecks, and a revised budget.

This window limits dependence on a single transient Q reading; it does not establish an equilibrium or validate the learned average policy. Continue logging Q and average-policy performance, game-length distribution, mutual-pass/round-limit endings, and draw rate after opening. Do not automatically clear the reservoir after a later bad read. If the optional whole-run zero-opening reference is used, it has no readiness check at time zero and reports these later diagnostics without rewriting its history.

Profile `H_N` using the selected Q reuse and target, including target inference, actor sampling, averaging updates, evaluation/checkpoint overhead, and measured memory limits. If the candidate record milestones cannot fit the recorded project ceiling, change the planned scope before training. The review reports Q-only 9×9 runtime, not complete NFSP runtime; this document supplies no validated full-learner forecast in its place.

### 6.4 Route-specific full-study ledger

Let `P` be actual PPO-route device-hours consumed, up to 72 for its complete A2 plan. If the matrix preflight fails, it normally stops within its 24-hour component instead of consuming all 72. Record actual abandoned-pilot costs separately when they cannot be reused.

The selected NFSP A2 ceiling is:

```text
N_A2 = 96 + 3*C1
       + I4 * 3*C4
       + IT * 3*CT
       + I0 * B_0
       + E
       + D
```

`I4`, `IT`, and `I0` select the additional reuse, Retrace, and zero-opening reference work. `E` sums selected stability-extension ceilings (each `3*CE`); `D` is zero unless the diagnostic is selected, then at most 12. Add the matched uniform comparison budget `B_U` to the campaign ledger, without double-counting reusable primary work. The Q-driven/5M configuration is already the primary allocation, not an extra `B_Q` variant. There is no `I7`/`CS7` trial. Do not select all optional work by default or omit failed-pilot costs.

The four-arm scientific study has twelve training runs, reusing completed base-arm training. Its scientific evaluation/probe-inference ceiling is 16 device-hours, with eight elapsed CPU hours for probe construction.

| Route/scenario | Full device-hour ceiling before separate reserves/promotion |
|---|---:|
| PPO only, **conditional on real preflight and A2 success** | `72 + 12*24 + 16 = 376` (about 15.7 continuous device-days) |
| NFSP after the selected PPO attempts | `P + N_A2 + 12*H_N + 16 + B_U` |
| Both routes explored for a stated comparison | Sum actual unreused selection/pilot costs, then the chosen study expression; no double-counting reusable base runs |

For an NFSP route, the scope decision must contain filled numerical values for `P`, `N_A2`, and `H_N` at the chosen replay ratio. Until those are profiled and entered, **the NFSP full campaign is uncosted and cannot be selected on the PPO figure**. Record the decision to stop, run a limited pilot, or take on the numerical full-study ceiling; there is no default launch of the large study.

The displayed NFSP expression already includes `B_U`. Add at most 24 device-hours if the corrective reserve is used, and eight per selected promotion round. CPU-only ceilings total 24 elapsed hours if all CPU stages are used. Add actual abandoned-pilot or optional-experiment costs that cannot be reused. Continuous device-days are just hours divided by 24, not a wall-time promise.

Match scientific arms at equal completed fresh-record exposure and a common selected learner/update/opening protocol. Scientific game evaluation remains 24,000 initially, at most 96,000 under the preplanned endpoint; promotion retains its separate calibrated sample plan. At milestones, inspect coverage, value errors, replay counts, label age, and deployed readiness. Do not convert continuing improvement at an artificially shortened cap into a completed algorithm failure or relax a gate to fit a convenient cost scenario.

## 7. Reproducibility and deliverables

Record the rule version, observation schema, action indexing, architecture, learner selection, initialization and sampling seeds, opponent schedules, critic buckets, optimizer configuration, learner decisions, joint rounds, all update counts, and compute usage. For the NFSP-style branch, additionally save Q/target parameters, actual replay actions/probabilities, coverage/actor-loss masks, frozen greedy reservoir targets and source versions, reservoir counters, actual mode-specific record counts, bounded-Q/target settings, loss-bearing replay ratio, fresh-record refresh schedules, reservoir-opening threshold/events, and the selected route/profile ledger. Retain sequence metadata when using or comparing sequence targets, and save opaque-state benchmark codebooks. Apply the retention classes below: archive network/evaluation artifacts separately from the bounded set of complete resume bundles. Retain deployment-mixture weights and the network versions they reference. Preserve exact seeds and implementation versions for rule benchmarks and tactical fixtures.

### Checkpoint retention and disk guard

Apply these rules on the **training-data filesystem on `home-ai`**, not merely the machine holding this Markdown file:

| Artifact class | Retention |
|---|---|
| Network/evaluation archive | Keep every declared milestone and every network version referenced by an opponent pool, mixture, evaluation, or publication. Store weights, schema/config, identifiers, and metrics permanently for the project; do not prune them to make replay room. |
| Active-run resume bundles | Keep only the latest **two successfully verified** complete bundles per active run. A bundle includes model/target/optimizer state, replay, reservoir and sampling counters, RNG states, and rollout/worker progress needed to resume. |
| Completed-run resume state | Keep one verified final complete bundle per completed run; retire its older bulky resume bundles after final verification. |
| Temporary/staging files | Account for their peak size. Remove only run-owned abandoned staging files after confirming that no live save uses them. |

Older weights remain usable for evaluation but do not promise full optimizer/replay resumption. Do not copy an entire replay/reservoir snapshot into every permanently retained network checkpoint.

Measure actual bundle sizes in the first save. Before each run, before every save, and periodically between saves, check free bytes on the destination filesystem. Use **20 GiB as a minimum free-space reserve** and require:

```text
free_bytes - estimated_peak_additional_write_bytes >= 20 GiB
```

The peak estimate includes the new temporary resume bundle, any simultaneous network archive writes, manifests, and compression/copy overhead; it cannot assume an old bundle has already been deleted. Also report retained archive/resume totals and expected next-save size. The review's approximate 123 GB free and 150 MB–1 GB snapshot estimate are not capacity guarantees.

At a consistent training boundary, write the new bundle to a temporary path on the same filesystem, verify its manifest and checksums, and flush it before atomically committing its name and latest-generation pointer. Only after commit and read-back verification may the oldest third resume bundle be removed. A failed write leaves the previous two valid bundles and permanent weights intact. Test interrupted saves, checksum failure, rotation, and restoration in Phase A.

If free space is already below reserve or the projected write would breach it, pause new rollouts/updates before starting the write and report the last durable checkpoint and the required space. Do not truncate a checkpoint, delete permanent model history, or delete unrelated files. Resume only after the capacity condition is satisfied. A completed-run finalization follows the same verified-save rule before pruning older heavy state.

### Deliverables

The first useful deliverable consists of:

- The versioned rules, deterministic resolver, and capture/ko/scoring regression fixtures.
- Solved matrix games, pure-action exploration-label controls, bounded Gray Go subgames, opaque-state delayed/branching controls, and tabular comparisons with an exact gate report.
- Random and tactical baselines with documented behavior.
- The 64-channel, eight-block policy architecture and the selected, verified trainer; NFSP is implemented/evaluated when the conditional fallback or comparison is selected, not automatically funded in full.
- A verified deployed actor, episode mixture, or NFSP average policy, with any distillation tested separately.
- Versioned training/evaluation pools, numerical promotion checks, and a budget report.
- The four-arm rule/exposure comparison, gray-lifetime measurements, and certified equilibrium-consistent collision probes, with incomplete controls identified explicitly.

The central research question is whether anticipation of simultaneous commitments produces robust mixed strategies and useful control of gray-stone formation. Operationalize it through the exposure-controlled comparisons and equilibrium-consistent probes in §§5.3–5.4, without inferring internal intent from collision counts. Claims of greater difficulty than ordinary Go or other variants require controlled comparisons.

## 8. Review disposition

All reviews below are preserved verbatim as historical records. Their simulation counts, board scores, toy-learning results, and throughput are reviewer-reported observations; this revision did not rerun the throwaway C++ resolver or the reviewers' neural-learning experiments. Measurements from uncapped games must not be treated as measurements of the adopted 200-round game. The current main text governs the design where it differs from an earlier recommendation.

The eighth review reports a Q-only 9×9 trainer with the planned board network and also toy chain/short-task experiments. It does not evaluate the complete average-policy learner or every required control. The reviewer reports scripts and logs at `home-ai:~/graygo-review`; this revision did not fetch or independently audit them. Phase A should check those artifacts' rule version, record/update counts, seeds, evaluation procedure, and job concurrency before treating the measurements as comparable baselines. Use unrounded log values for threshold decisions; a rounded value of `0.050` does not establish a `0.05` pass.

| Comment | Disposition |
|---|---|
| 1. Capture ambiguity | Preserve the explicit capture-first owner decision already incorporated in §1.2. |
| 2. Endless cycles | Preserve two-round ko and the owner-selected 200-round scored limit. Do not adopt the review's alternative full-game history rule. |
| 3. Ko targets the original taker | Document the consequence and require the outgoing-move regression fixture; preserve the supplied ko rule. |
| 4. Gray's strategic role | Add matched rule ablation, gray persistence/scoring metrics, and controlled action-value probes. Qualify the proposed intent inference. |
| 5. Latest versus averaged policy | Add a solved-game gate and explicit mixtures. The second revision budgets NFSP alongside PPO; distillation is only compression of a passing strategy. No convergence guarantee is asserted for either neural learner. |
| 6. Opponent-conditioned critic | Adopt training-only bucket conditioning; keep search evaluation separately calibrated. A self-play bucket does not supply minimax values. |
| 7. Numerical promotion | Add confidence-aware thresholds, limited to symmetric starts and the tested attack set. The second revision widens the declared regression tolerance and requires full-gate power calibration before use. |
| 8. Budget and scope | Add staged decision/device budgets and stop rules. Replace redundant symmetry experiments with tests and simplify the no-change restriction using its equivalence proof. |
| 9. Else constraint | Explicitly drop it from the AI design while retaining top-k-plus-tail explanation output. |

### Second-review disposition

| Comment | Change and qualification |
|---|---|
| 1. Only six A2 PPO updates | Give PPO its own 1,024-record rollout and 2M-record A2 cap. The third revision separately enlarges NFSP and adds a fifth, long-horizon task family. |
| 2. No working fallback | Budget and specify NFSP alongside PPO, plus one diagnostic reserve. Test the real trainers; do not treat the review's toy results as proof of failure or neural NFSP as guaranteed convergence. |
| 3. Promotion too underpowered | Change per-opponent regression tolerance to `-0.15`, explicitly acknowledging the larger allowed tradeoff. Calibrate the aggregate, regression, and absolute-floor conditions together; allow a preplanned 8,000-game endpoint. Do not adopt illustrative pass rates as measured facts. |
| 4. Fixed-baseline bias | PPO's fixed-baseline share reaches zero by 5M decisions; A2 uses the final zero-anchor schedule. NFSP uses its native procedure. A smaller imposed baseline share is not a universal equilibrium guarantee. |
| 5. Endgame resolution | Add six precisely defined countdown-threshold flags while retaining the normalized scalar. This is a cheap feature-design choice, not a measured finding that the original network cannot resolve a 0.005 difference. |
| 6. Gray research lacks decisions/budget | Add endpoints, matched seeds and four exposure-controlled arms. The fifth revision provides complete path totals and requires a freshly profiled NFSP per-seed cap rather than relying on the old fixed estimate. |

### Third-review disposition

| Comment | Change and qualification |
|---|---|
| 1. NFSP allocation/counters | Raise NFSP A2 to 15M aggregate records per task family and seed, add tabular controls and define shared versus per-role update counters. The fourth revision separates short-task and target-comparison time allocations. Reviewer toy success is not a replicated guarantee. PPO uses a proposed small entropy floor to be evaluated in A2. |
| 2. Delayed reward propagation | Add 50/100/200-round tests. The fourth revision removes answer-revealing adapters from the gate and selects one-step versus Retrace by an explicit comparison; target refresh is held constant. |
| 3. Hidden-opponent behavioral test | Replace fixed-opponent best-response grading with complete-game value gaps for both learned roles and certified collision ranges across admissible near-equilibria. Avoid arbitrarily selecting one equilibrium when many exist. |
| 4. Gray exposure confound | Add matched gray-start training to both rule arms and retain the original comparison, giving a four-arm study. A disappearing effect is consistent with exposure/curriculum confounding, not proof that gray can never matter. Fund the additional controls explicitly. |

### Fourth-review disposition

| Comment | Change and qualification |
|---|---|
| 1. Exploration in the average policy | Verify the repeated-state epsilon arithmetic, then use frozen greedy target distributions at every best-response-mode visit while retaining actual exploratory actions in RL replay. This is an NFSP-style variant; changed state visitation and imperfect greedy labels still require tests. |
| 2. Chain shortcut and weak traces | Require opaque state identities, multiple action-equivalent choices, real behavior modes, low-overlap fixtures and a meaningful-branch control. Give PPO's harder chain a separate larger cap. Compare one-step and Retrace; raw n-step is diagnostic only. Nonzero epsilon generally shrinks traces rather than making all nongreedy coefficients exactly zero. |
| 3. NFSP-specific 9×9 milestones | Keep separate 20M screening/80M confirmation proposals and both-policy diagnostics. The fifth revision makes their time ceilings contingent on profiling the increased Q replay work. |

### Fifth-review disposition

| Comment | Change and qualification |
|---|---|
| 1. Avoided branch lacks data | Add value-only forced-root coverage episodes, including the actual root-action edge for Q. Exclude them from PPO actor loss and reservoir labels, and exclude forced root returns from PPO critic fitting. Count the extra data and report branch coverage. |
| 2. Averaging lag is misgraded as delayed credit | Short games still grade the deployed average policy; the long family grades Q/greedy policy and value propagation. Average-policy progress remains visible and mandatory at 9×9. Burn-in is considered as an untested optional protocol, not silently adopted or used to erase history. |
| 3. Unbounded/undersampled Q | Bound action values and project loss targets to `[-1,1]`. Raise default loss-bearing Q reuse from `0.125` to `1`, compare `4`, and keep target refresh fixed by fresh records. Increase the conditional target-comparison allocation and retire unprofiled NFSP runtime estimates. Bounds alone do not solve data starvation or bias. |
| 4. Cost and mandatory dual learning | Keep cheap-first selection. The sixth revision makes the PPO cost explicitly conditional on real-trainer preflight and requires filled NFSP costs at the selected reuse before the A2-exit decision. No training has been launched. |

### Sixth-review disposition

| Comment | Change and qualification |
|---|---|
| 1. PPO route/cost may not apply | Add a real-trainer matrix preflight comparing the taper and one fixed higher-entropy setting with the same KL stop. Retire the PPO budget as the planning basis if that route fails; reviewer toys remain evidence rather than a universal impossibility result. |
| 2. Reuse comparison cut short | Test one-step reuse first, giving `R=4` the same 15M-record cap per seed. Enter time ceilings from profiles. Retrace is conditional on one-step results, not an automatic expensive first comparison. |
| 3. Deployment averaging decision too late | Move a zero-versus-5M reservoir-opening comparison into A2 when needed, with defined pre-opening behavior and no retrospective clearing. Retain short-game thresholds and add separate long-chain average-policy readiness bands before 9×9 learning. Neither early opening nor a passing greedy Q policy guarantees deployed strength. |

### Seventh-review disposition

| Comment | Change and qualification |
|---|---|
| 1. Opening sensitivity and near-pass handling | Retain the marginal-window rule. The eighth revision retires the 7M fallback, promotes Q-driven pre-opening to primary, and replaces the single opening read with a fixed three-checkpoint decision. |
| 2. Replay retention and actual machine | Record the review-reported `home-ai` hardware and require live verification. Preserve network archives, retain two verified active resume bundles and one final bundle per completed run, and enforce a projected 20 GiB reserve with atomic saves. Do not turn the review's rough throughput or free-space figures into measured guarantees. |

### Eighth-review disposition (2026-11-01)

| Comment | Change and qualification |
|---|---|
| 1. Primary route fails the reported chain/9×9 combination | Set Q-driven pre-opening at 5M and a 12,800-fresh-Q-record multiround refresh as the primary configuration. Keep uniform matched as a comparison. Recheck bounded subgames and meaningful branches; Q-only wins do not certify the deployed average policy. |
| 2. Later opening does not repair the cause | Remove the 7M fallback and its allocation rather than extending it to 17M. The unrelated, tightly defined marginal stability extension remains; it does not move opening. |
| 3. Single opening read is unstable | Use the fixed 4.5M/4.75M/5M Q snapshots and require two complete passes, with confidence coverage across snapshots, opponents, and planned looks. Do not cherry-pick snapshots or roll the window until success. |
| Evidence and runtime limits | Retain all reported results as reviewer evidence, record the remote artifact location, and require full-learner profiling. Separate shared-device throughput from per-job elapsed time; log short self-play games without assuming they are failures. |

**Next planned step:** Phase A implementation, correctness checks, storage validation, and profiling, followed by the declared real-trainer gates. The remaining learning/runtime claims require verification. This document update does not start implementation or training.


---

## Review comments (Claude, 2026-10-31)

**Verdict:** the training plan is sound and the order of work is right. Two rule gaps need the owner's decision before anyone writes the resolver, because the resolver is the foundation of everything else. The plan also has no test for its central research question.

To check the rules, I wrote a throwaway C++ resolver and ran scripted boards and random-agent games. "Eye-preserving" below means an agent that plays uniformly at random, never fills its own one-point eye, and passes only when it has nothing else to play. These agents show what the rules allow. They say nothing about how trained play will behave.

### Decide before writing the resolver

**1. The capture phase is never defined (§1.2).** *Resolved: capture-first; see Owner decisions below.* The rule says captures precede suicide, but nothing says which dead groups count as captured. No implementer can derive it, and the two natural readings give different results:

```text
        c2  c3  c4  c5
  r2     .   W   B   .
  r3     W   p   W   B
  r4     W   B   B   W
  r5     .   q   W   .

Black plays p; white plays q. All other intersections are empty.
```

The white stone at r3c4 has only liberty p. Black's group, including p, has only liberty q. Both are dead after placement.

| Reading | Result | Score on this board |
|---|---|---|
| **Capture-first:** remove every dead group containing no stone placed this round, recompute, then remove whatever is still dead | White's stone goes and black's group lives through the new liberty | 40.5–40.5 |
| **Mutual-death:** the literal "if both groups are dead, both are suicidal" | Both groups go | 38.5–42.5 |

If white passes instead, both readings give an ordinary capture, so this is purely a simultaneous-move question. The readings disagreed in about 0.2 rounds per 1000 of eye-preserving play, or roughly one game in 60. Tactical agents may reach this shape more often.

I recommend capture-first. It is one sentence and independent of removal order, and it treats p like any other capturing stone. But the owner's quote can be read the other way, so this is the owner's call. Under capture-first, the §1.2 example resolves in the capture phase rather than the suicide phase, with the same result. Its "if neither is rescued" wording should go too, because only groups containing a new stone can be rescued.

**2. Two-round ko allows endless games, and this is more than an infrastructure concern (§1.4, §1.6).** *Resolved: two-round ko kept, with a 200-round limit; see Owner decisions below.* A four-round double-ko cycle never matches the two saved boards, so each new board clears the restrictions. I scripted that cycle for 100,000 rounds: it returned to the start every four rounds with zero forbidden points. Eye-preserving agents also fall into endless play with no intent: 10 of 5,000 9×9 games were still running at the 100,000-round cap. Every one I sampled (8 of the 10) had settled on exactly 8 distinct boards, consistent with three independent ko flips.

Better groups should make this more likely, not less, because the shapes that sustain these cycles are live groups. It then becomes a training hazard. With discount 1.0 and bootstrapping at chunk boundaries (§4 Phase B), a cycle state has no terminal value to anchor its estimate. A losing player can learn that a cycle avoids −1. Calling these games "unresolved" reports the problem but gives no utility to train on, and §2 and the Phase E matrix game both assume one exists.

I recommend comparing against every earlier board in the game instead of the last two, keeping the same restriction mechanism:

| 9×9, eye-preserving, 5,000 games | Two-round history | Full-game history |
|---|---|---|
| Still running at 100,000 rounds | 10 | 0 |
| Median rounds (finished games) | 87 | 86 |
| p99 rounds | 343 | 105 |
| Longest finished game | 1,949 | 129 |
| Double-ko cycle | Runs indefinitely | Blocked at round 5 |

This also makes termination provable:

1. Restrictions clear only on a never-seen board, and boards are finite, so clearing eventually stops.
2. After that, restrictions only grow, so they eventually stop growing too.
3. From then on, returning to a board forbids any placement previously made from it, which would be new growth. So a board that recurs can only have been left by mutual pass, and that ends the game.

The cost is that §3.2's "complete rule state" stops being true, because the network no longer sees the full history. Superko Go programs accept the same compromise, and the forbidden-point channels still carry the legal consequences. Most of the unresolved-game handling in §1.6 and §4 then goes away. If the owner keeps two-round ko, the rules need an explicit outcome for repetition; a referee timeout is not enough.

**3. Confirm this ko consequence is intended.** The restriction lands on the player who took first. In my test, white takes, black retakes at once, and white is the one forbidden; black has no restriction. That reverses ordinary Go, where the retaker needs a threat. It follows faithfully from §1.4, so the point is only to confirm the intent and pin it with a test.

### Training design

**4. The central research question has no test (closing paragraph of §7).** Nothing in §5 or §6 can show "deliberate control of gray-stone formation", and the baseline numbers suggest it may have little to work with. Eye-preserving games average 3 collisions, but only 0.2% of finished games end with any gray stone. Those agents wipe out one colour in 54% of finished games, so trained play may differ a lot.

There is also a structural reason for the result. Gray grows only through further collisions, so it can hardly build eyes, and under area scoring, filling your own territory to capture it costs nothing. If trained games look the same, gray is a lost tempo plus a brief liberty block, and the fractional rows in §1.5 rarely apply. The owner would want to know that as a game-design result. I would add two experiments:

- **Rule ablation:** train the same setup with a collision placing nothing. If strength and style barely change, gray is not doing strategic work.
- **Intent:** across positions, compare the collision probability `sum_a pi_B(a) * pi_W(a)` with the value of a collision, `Q[a,a]`, against other outcomes. Control means the overlap rises when a collision favours the player.

Also report gray survival to the end of the game, for trained agents.

**5. Make the PPO mixing question a gate (§3.4, §4 Phase B).** I did not measure this one. Gradient self-play in zero-sum simultaneous games is known to cycle, with the time average approaching equilibrium rather than the latest iterate (for example, Mertikopoulos et al., "Cycles in Adversarial Regularized Learning"). The entropy bonus damps the cycling but biases the result toward uniform. The plan evaluates and ships a single checkpoint, which may be a point on that cycle rather than the mixed strategy the project is looking for. The opponent pool helps robustness, but it doesn't produce an average policy.

Phase A already builds solved subgames, and "equilibrium gap" is only a metric in §5. Make it a gate: run the actual trainer on those subgames before spending on 9×9, and measure the gap for both the last checkpoint and the average policy. If only the average converges, average-policy distillation (or the NFSP branch) moves from optional into the first deliverable.

**6. Give the critic the opponent during training (§3.5, Phase E).** With a 40/40/20 pool and the opponent hidden, `V` is an average across very different opponents. That adds noise to the advantages, and Phase E then solves a matrix over leaf values that reflect the pool rather than good play. Feed the critic an opponent bucket during training and never give it to the actor. Search can use a fixed self-play bucket. This doesn't conflict with the §4 Phase D no-leak rule, which concerns deployed inputs, and it is cheap to do from the start.

**7. Turn promotion into a number (§5).** The game is symmetric, so its value is exactly 0. An expected result below 0 against any opponent therefore proves exploitability, no rating needed. The worst entry in a candidate's cross-play row, with its uncertainty, can serve as the concrete meaning of "no substantial newly exposed weakness". The pool remains an attack set, so a good number is not proof of strength.

### Scope

**8. Add a budget and a stop rule; cut what can't change a decision.** §6 implies about 8 distinct training configurations at 3 seeds each, so 24 runs before any search work. The doc gives no run length, hardware, or stopping condition. Phase B needs a stated budget and a kill criterion, such as a required margin over the tactical baseline within a set number of decisions, or stop and diagnose. Three things can go:

- **Zero-padding arm (§6.1, §7):** nobody would ship it on a torus, so it can't change a decision.
- **Translation augmentation (§3.7):** a circular CNN with average pooling is already exactly translation-equivariant, so this does nothing. Replace it with a unit test and keep only rotations and reflections. This also means the empty-board policy is necessarily uniform across all 81 points, which is fine for this game.
- **The "qualifying suicidal move" predicate (§1.2, §1.3):** in a no-change round every placement has vanished, and one connected to a friendly group would have taken that group with it, changing the board. So the rule is simply "in a no-change round, forbid every placement". That also avoids defining "friendly" for a gray stone.

**9. Say explicitly that the else-branch constraint is dropped (§4 Phase D).** The earlier notes made it a designer-proposed constraint and a training objective, and this doc reduces it to a display quantity in passing. That is the right call: a full distribution is already total, and a finite game doesn't need the constraint for an equilibrium to exist. Conditional replies also can't execute within one simultaneous round. But since it was the designer's idea, it deserves a stated decision and reason.

### What held up

- **Sparse terminal rewards (§4 Phase B).** I expected untrained games to run so long that rewards never arrived, and that was wrong. Uniform random agents finished 200 of 200 9×9 games, with a median of 347 rounds and a maximum of 2,356. No shaping is needed at the start, as the doc says.
- **Stalling.** A never-pass player against an eye-preserving player finished 500 of 500 games, with a median of 95 rounds and a maximum of 286. Endless play needs a ko cycle; refusing to pass isn't enough.
- **Resolver throughput.** The unoptimized single-threaded resolver ran about 400,000 rounds per second, including random move selection. In a compiled language, network inference should be the bottleneck, not the rules.
- **The §1.2 example** resolves as described under both readings.
- **§2's corrections to the earlier notes** are right: independent draws give a product distribution, and expected value is taken over the whole prediction rather than a sampled guess. So is the §3.2 rule against masking possibly suicidal moves.

### Owner decisions (2026-10-31)

Both are now written into the rules above.

- **Capture phase (comment 1):** in the example, white's r3c4 stone is removed first and black's group lives. §1.2 now states the capture and suicide phases as an explicit order.
- **Ko and game length (comment 2):** two-round ko stays, and play is limited to 200 rounds. A forced ending gives every game a real result, so the training hazard from endless games is gone. §1.6, §2, §3.2, §4 and §5 are updated, including a new rounds-remaining input.

Where the decision left details open, I wrote these in; the owner should correct any that are wrong:

- The limit counts rounds, not individual stones.
- A forced end is scored by §1.5 as the board stands.
- The limit applies to 9×9 only for now.

Measured effect of the limit on random 9×9 agents:

| Agents | Games reaching round 200 |
|---|---|
| Uniform vs uniform | 70.8% |
| Eye-preserving vs eye-preserving | 1.7% |
| Eye-preserving vs never-pass | 1.6% |
| Never-pass vs never-pass | 36.3% |

Most early training games will therefore be ended by the limit, and their results will be close to noise until the policy learns to protect its groups. That is expected and needs no special handling.

The main consequence to watch is that the last rounds now matter a lot, as §1.6 describes: the scripted 45–36 lead flips to 36–45 on final-round drops. I don't think this needs a rule change, since filling territory costs no points and defends against it. But a player who is behind will try to reach the limit, so strong play may often run to round 200. The final-rounds margin metric in §5 will show whether the limit, rather than the middle game, is deciding results.

---

## Second review (Claude, 2026-10-31)

This reviews the revision containing §8, and reports medium or higher issues only. I found six: three high and three medium. Four are in the new gates: the first two would stop the plan at Phase A2, and the promotion rule would almost never pass. The rules in §1 hold up, including the suicide equivalence proof, and so do the gap formulas and the critic separation.

### High

**1. The A2 budget allows only about 6 PPO updates (§4 Phase A2, §6).**

- **Problem:** 200,000 decisions at Phase B's rollout of 32,768 is about 6 updates, so the "final five scheduled evaluations" would cover nearly the whole run. In a toy PPO I wrote using the doc's settings, on an asymmetric 2×2 game, the gap after 6 updates was 0.13–0.27, against a gate of 0.05.
- **Why a smaller rollout isn't enough:** it weakens §5.1 rule 1, which says the candidate's training configuration passed A2.
- **Fix:** give A2 its own stated settings and a much larger cap, and reword rule 1 to say the gate validates the algorithm, not the batch settings.

**2. The gate will probably fail, and the plan has no working next step (§4 Phase A2, §6 item 2).** More budget didn't help the toy, and no run reached 0.05:

| Rollout | Decisions | Updates | Latest policy gap | Mixture gap |
|---|---:|---:|---:|---:|
| 32,768 | 2M | 61 | 0.30–0.82 | 0.08–0.18 |
| 32,768 | 20M | 610 | 0.56–1.14 | 0.27–0.28 |
| 4,096 | 20M | 4,882 | 1.15 | 0.14–0.45 |
| 1,024 | 2M | 1,953 | 0.73–1.15 | 0.24–0.40 |

Each row covers three seeds. The mixture is uniform over the last half of the per-update checkpoints.

- **What it shows:** the latest policy gets worse with training, ending near a pure strategy, and the mixture doesn't reliably improve.
- **Caveat:** this is my own small network on one game with no opponent pool, so it's evidence, not proof.
- **Gap in the plan:** "stop and diagnose" has no next step. A distilled average can't be expected to beat the mixture it copies, and NFSP has no budget.
- **Fix:** run the toy test on the real trainer first. Also put a second learner into A2 from the start, such as NFSP or another method built to converge in these games, and budget for it.

**3. Promotion rule 3 almost never passes (§5.1).** It's labelled as a −0.05 non-regression tolerance, but simultaneous confidence intervals turn it into required improvement. With 10 close matchups and 2,000 games, the interval half-width is about 0.10, so a candidate must show about +0.05 against every close opponent. That assumes few draws and that pairing doesn't reduce the variance much.

| True improvement against every opponent | Pass rate at 2,000 games | Pass rate at 8,000 games |
|---|---:|---:|
| +0.03 | 0.0% | 70% |
| +0.10 | 50% | 100% |

This matters more in a game with nontransitive matchups, where a better candidate will usually slip against someone. Keep 2,000 games and widen the tolerance to about −0.15. By my calculation, that passes a uniform +0.03 candidate about 93% of the time and still rejects a true −0.15 regression against any one opponent about 99.9% of the time.

### Medium

**4. The 20% fixed-baseline share can fail the gate by itself (§4 Phase C, Phase A2).** A2 tests the actual opponent scheduler, but a fixed baseline moves the learner's stable point away from equilibrium. On the same 2×2 game, with a uniform baseline, the exact gap at that point is:

| Baseline share | Gap at the stable point |
|---:|---:|
| 20% | 0.074 (fails) |
| 10% | 0.033 |
| 5% | 0.016 |

Reduce the share to 10% or less by late training, and run the gate on that final schedule. The same bias also applies at 9×9, where the policy would lean toward exploiting the baselines.

**5. Rounds remaining is hard for the network to read near the end (§3.2).** Normalized by 200, one round left and two rounds left differ by only 0.005, yet §1.6 makes those last rounds tactically distinct. I didn't measure this, but the fix is cheap: keep the normalized value and add flags for 1, 2, 3, 5, 10 and 25 rounds remaining, or similar.

**6. The central research question still has no pass/fail rule or budget (§5.3, §6).** §5.3 now has the right experiments, but it never says what result would count as "gray matters" or "purposeful control", while almost everything else in the doc has a threshold. The budget paragraph in §6 also covers only optional comparisons, even though this one is required and matched training could cost about 72 device-hours (three seeds at 24 hours each). Before running it, set the budget and a decision rule, such as:

- **Gray matters:** the base-trained policy beats the variant-trained policy under base rules by a stated margin.
- **Purposeful control:** collision probability moves the right way on a stated share of the tactical probes.

---

## Third review (Claude, 2026-10-31)

This reviews the revision that adds NFSP, the A2 settings and §5.4, and reports medium or higher issues only. I found four: two high and two medium. All four are in the new material. The earlier fixes are otherwise in good shape, and the second-review items are properly addressed.

### High

**1. NFSP is the fallback, but its A2 budget looks too small to pass (§4 Phase A2, §4 Phase B2, §6).** I ran toy versions of both learners at the doc's A2 defaults on two gate tasks. The table shows the worst exact gap over the final five evaluations, against a gate of 0.05:

| Learner | Asymmetric 2×2 | Rock-paper-scissors |
|---|---:|---:|
| PPO latest, 2M decisions | 0.85–1.15 | 1.00 |
| PPO episode mixture, 2M | 0.18–0.85 | 0.50–1.00 |
| NFSP average policy, 2M | 0.19–0.29 | 0.44–0.49 |
| NFSP with an exact best response in place of Q learning, 2M | 0.02–0.04 | 0.08–0.09 |
| NFSP average policy, 10M | 0.02–0.03 | 0.04–0.05 |

PPO rows combine runs with and without the KL stop. The 10M row used two seeds and the others three. I counted 2M decisions as 1M per player, with one update per 128 of that player's own decisions; counting decisions across both players would double the updates, which I didn't test. These are compact toy networks, so this is evidence, not proof.

- **What it shows:** NFSP looks viable, but it needs about five times the budget. PPO's policies collapsed to near-pure strategies, and with entropy at zero for the second half of the run, a collapsed policy has nothing to pull it back.
- **Why it matters:** as written, both learners would probably fail, use up the single corrective rerun, and end in the "blocked" outcome over a budget number.
- **Fix:** raise NFSP's A2 budget to at least 10M decisions per task and seed before the first run, and recheck the 24 device-hour cap. Rock-paper-scissors only just clears at 10M, so more margin would help. Include a tabular NFSP control on the same schedule, so a budget shortfall can be told apart from a bug.

**2. NFSP's one-step targets may not carry the result back through a 9×9 game in budget, and A2 can't detect this (§4 Phase B2, §6).** The 9×9 settings use terminal-only rewards, discount 1.0, one-step Double-DQN targets, one Q update per 512 decisions and a target refresh every 200 updates. That is one refresh per 102,400 decisions:

| Stage | Learner decisions | Target refreshes |
|---|---:|---:|
| Smoke | 1M | about 10 |
| Screening | 5M | about 49 |
| Confirmatory cap | 20M | about 195 |

Each refresh moves value information back by about one round, while games can run to 200 rounds. Generalization across similar boards will help, but I didn't measure how much. At the screening gate, early-game Q values may still reflect little about final outcomes. PPO doesn't share this limit, because GAE at 0.95 carries returns about 20 rounds in one update, so the two branches aren't on equal terms.

A2 can't reveal any of this, because its tasks are one-step or short-horizon. A learner can pass A2 and then fail 9×9 screening, which sends the diagnosis toward the wrong cause.

- **Fix:** use n-step or Q(λ) targets for the 9×9 branch, and add a long-horizon solved or chain-style task to A2 so selection tests how far terminal rewards propagate.

### Medium

**3. The behavioral probes require best responses to an opponent the actor can't see (§5.4).** A family passes only if the policy's utility is within 0.05 of the best-response value against a declared opponent distribution, but that distribution isn't an actor input.

- If the declared distribution is the opponent's equilibrium, a mixed equilibrium player is indifferent across its support, so near-optimal play doesn't require any particular collision probability, and the family fails the eligibility rule.
- If it isn't the equilibrium, a mixed equilibrium policy generally isn't a best response to it, so the policy the project is looking for can fail the utility condition.
- What's left is mainly families where one action dominates whatever the opponent does. Those test tactics, not mixed-strategy control.

So the 24-of-32 criterion can either fail good mixed policies or pass by being limited to tactics. **Fix:** drop the declared opponent. Build pairs of states whose solved equilibria differ in `P_collision` by at least 0.10, and require the policy's own `pi_B * pi_W` collision probability to shift in that direction while its exact gap stays within 0.05 in both states. Any adaptive test can stay separate, with evidence delivered through public history.

**4. The performance test can pass because of unfamiliar inputs, without gray having strategic value (§5.4).** `P_E` is trained in a world where gray stones never appear, so its gray input channels are untrained when it plays under base rules. `P_G` is also off-distribution under the variant. Each policy is therefore likely to do worse under the other's rules, which raises both `u_G` and the interaction `u_G - u_E`, whether or not gray matters strategically. The interaction term guards against one run simply being stronger, but not against this.

**Fix:** place a few random gray stones in the starting positions used for variant training, so that `P_E` has learned how gray stones capture, block and score, and differs only in never seeing a collision create one. Report the original comparison beside this one. If the effect disappears with seeding, the original result was about unfamiliarity.

---

## Fourth review (Claude, 2026-10-31)

This reviews the revision that adds the delayed-reward family, Retrace targets, the four-arm gray study and the certified probes, and reports medium or higher issues only. I found three: two high and one medium. All three concern the NFSP branch and its gates. The rest of the new material holds up, including the Retrace formula, the budget totals, the four-arm design and the probe rules.

### High

**1. NFSP can't reach the 0.05 gap on some pure-solution tasks, whatever the budget (§4 Phase B2, A2 acceptance, §5.4).** §4 Phase B2 stores best-response-mode actions in the reservoir, and those include epsilon-greedy exploration moves. The reservoir samples the whole run, so the average policy inherits the run's mean epsilon of 0.0325 as mass spread across legal moves. Evaluation turns off forced exploration, but the trained average policy still contains it.

I simulated a perfect best response with this epsilon schedule, on a position where one action decides the game, so any other action swings the result from win to loss (2.0 utility):

| Legal actions | Mass on non-best actions | Gap | Against the 0.05 gate |
|---:|---:|---:|---|
| 2 | 1.6% | 0.033 | Passes |
| 3 | 2.2% | 0.043 | Passes |
| 5 | 2.6% | 0.052 | Fails |
| 10 | 2.9% | 0.059 | Fails |
| 82 | 3.2% | 0.064 | Fails |

This is the worst case, where every other action loses outright, and many positions will be milder. But such positions are normal near the end of a game, and they affect the A2 task with a pure solution and the §5.4 probes. The table also assumes a perfect best response, so real runs would do worse. A deployed average policy would also play about 3% near-random moves every round, which at 100 rounds is about three per game.

- **Fix:** store only non-exploratory actions in the reservoir, or store the greedy action in their place. In the same simulation, that brings the mass and the gap to zero. Because this changes the published NFSP recipe, include it in the tabular control too.

**2. The delayed-reward gate can pass without testing delayed credit (§4 Phase A2, Phase B2).** I found two blind spots:

- **The adapter can give the answer away.** The family allows "a compact observation adapter". If its input shows which branch the episode took, the value is just that flag, and a network can generalize from late states to early ones with no propagation at all. In a toy PPO at the doc's chain settings (2M records, 61 updates), the choice of input decided the result:

| Observation | Value error, 50 rounds | 100 rounds | 200 rounds | Gate of 0.10 |
|---|---:|---:|---:|---|
| Position and branch flag visible | 0.002–0.017 | 0.004–0.014 | 0.009–0.021 | Passes |
| Random fixed code for each state | 0.072–0.077 | 0.087–0.119 | 0.240–0.338 | Fails |

The policy gap was 0.021–0.025 in both cases, so only the value check separates them.

- **Forced moves hide trace cutting.** With one legal action, the Retrace weight never cuts, so the effective trace is 16.1 rounds. Real 9×9 play is different. The trace is cut whenever the logged action isn't the current greedy action, and 90% of episodes are average-policy mode. With 40 legal actions, the calculated effective trace is:

| Sequence type | Condition | Effective trace (rounds) |
|---|---|---:|
| Average-policy mode | Greedy action has probability 0.10 | 1.12 |
| Average-policy mode | Greedy action has probability 0.30 | 1.42 |
| Average-policy mode | Greedy action has probability 0.50 | 1.94 |
| Best-response mode | Greedy action changed for 10% of logged actions | 6.85 |
| Best-response mode | Greedy action changed for 30% of logged actions | 2.99 |

So most sequences behave nearly one-step in real play, while the gate sees 16 rounds. Faster target refresh is doing most of the work: about 195 refreshes by 5M records and 781 by 20M, compared with 49 and 195 before. Sequence replay and logged behavior probabilities are a real implementation cost, and they may add little.

- **Fix:** specify the chain's observation so that it hides the outcome, such as a random fixed code per state. Give each step many equally valued legal actions, and run it in the real mode mix. Compare Retrace, one-step targets and an uncorrected n-step target on that chain, with the tabular control. If Retrace doesn't beat one-step, remove it. Expect PPO to need more than 2M records on the harder chain, so set its budget before the first run.

### Medium

**3. NFSP's 9×9 milestones are still set for PPO (§6).** A2 now gives NFSP 7.5 times PPO's budget because a three-action game may need it, yet the 9×9 stages use the same 1M, 5M and 20M caps for both learners. Those stages judge the average policy, which is built from far less data than the record counts suggest:

| Stage | Records | Best-response records (10%) | Supervised updates |
|---|---:|---:|---:|
| Smoke | 1M | 100,000 | 1,953 |
| Screening | 5M | 500,000 | 9,766 |
| Confirmatory cap | 20M | 2,000,000 | 39,062 |

The reservoir also weights the earliest, weakest best responses the same as recent ones. I didn't run this at 9×9, so it's a judgment from the arithmetic. If so, NFSP could fail screening because of the schedule, and that would look like a strength problem. This matters because NFSP is the likely choice if PPO fails A2.

- **Fix:** set NFSP's own 9×9 milestones before the first run, report the best-response network's strength beside the average policy's at each one, and include 9×9 cost in the learner selection rule.

### Checked and not an issue

- **PPO on the chain's budget:** I expected 61 updates to be too few, but with the plain adapter it passed in all three seeds. The harder chain above is what fails.
- **Third-review refresh concern:** the new cadence largely addresses it on arithmetic, as above.

---

## Fifth review (Claude, 2026-10-31)

This reviews the revision that adds greedy-label averaging, the opaque-state chain, target selection and learner-specific 9×9 milestones, and reports medium or higher issues only. I found four: three high and one medium. The three high issues are all in the opaque-state delayed-reward gate, and in my toy runs neither learner passes it as specified. Each has a specific fix, and two of them are about how the gate is graded, not about how well the learners train.

I tested the gate with compact toy learners at the doc's settings. They use one-hot opaque states, a 256-unit network and 40 equivalent actions. The average policy is uniform on intermediate states, and its root-state labels are kept in a table. Real trainers may differ, so treat these results as evidence, not proof.

### High

**1. The gate requires accurate values on a branch that a good policy stops visiting (§4 Phase A2).** "Evaluate states from both outcome branches" conflicts with the 0.05 policy gap. A gap of 0.05 means the unfavorable branch gets at most 2.5% of episodes, and its share keeps shrinking as the policy improves. So the value error there can rise because the policy got better:

| PPO, 200-round chain | Policy gap | Favorable-branch error | Unfavorable-branch error |
|---|---:|---:|---:|
| Seed 0, 2.9M records | 0.010 | 0.000 | 0.031 |
| Seed 0, 5.9M records | 0.002 | 0.000 | 0.075 |
| Seed 0, 8.8M records | 0.001 | 0.000 | 0.447 |
| All three seeds, 10M records | 0.001 | 0.000 | 0.28–0.76 |

PPO would fail its gate on this check alone, even with almost no policy gap. NFSP shows the same pattern: with clipped targets, favorable-branch error is at most 0.055, while unfavorable-branch error is 0.39–1.79 across horizons in the 14M runs in comment 3.

- **Fix:** keep the both-branch check, but give it the data it needs. Start a fixed share of chain episodes, such as 5–10%, directly on each branch. Use them for value learning only, and exclude them from PPO's actor update and from the reservoir. Report coverage alongside the error.

**2. NFSP's average-policy gap on the long chain is limited by labels from before Q has learned (§4 Phase A2, Phase B2).** Greedy labels are frozen and the reservoir samples the whole run, so labels from before Q reaches the root stay in it. If those labels are about 50/50 for a share `t` of the run, the policy gap at the root is about `t`, because a wrong action costs 2.0 utility.

Meeting 0.05 would therefore need correct root values within the first 5% of the run, which is 750,000 of 15M records. At one refresh per 25,600 records, that's only 29 target refreshes to cover 200 rounds, with a measured trace of about 1.1 rounds in this toy. So this limit comes from the design, not from the toy's quality. In the runs in comment 3, the gap at 200 rounds was 0.28–0.94. One Retrace seed had a gap of only 0.001 because its root happened to favor the right action from early on, so a low gap here can also be luck.

- **Fix:** don't grade this family on the average policy. Grade Q and its greedy policy for delayed credit, and leave averaging to the short tasks. At 9×9, consider a declared burn-in where the reservoir opens only after Q passes a progress check, because the same early labels stay in the deployed policy there. I haven't tested burn-in.

**3. Q targets are unbounded and undersampled, so neither target candidate passes (§4 Phase B2).** With discount 1.0, maximizing over noisy actions adds upward bias at each step, and the bias grows with horizon. All values here lie within ±1, yet the doc's settings produced errors well above 1.0. At those settings, Q also trains on only 64 transitions per 512 new records, a replay ratio of 0.125, where PPO's is 4.

| One-step Q on the opaque chain | Records | Favorable error, 200 rounds | Root error, 50 rounds | 100 rounds | 200 rounds | Policy gap, 200 rounds |
|---|---:|---:|---:|---:|---:|---:|
| Doc settings | 14M | 2.26–2.43 | 0.44–0.85 | 1.18–1.50 | 2.23–2.98 | 0.49–0.94 |
| Targets clipped to ±1 | 14M | 0.007–0.015 | 0.39–0.43 | 0.88–1.03 | 1.70–1.75 | 0.77–0.86 |
| Clipped, 512 transitions per 512 records | 14M | 0.000–0.007 | 0.27–0.35 | 0.79–0.91 | 1.79–2.00 | 0.64–0.67 |
| Clipped, 4 × 512 transitions per 512 records | 9M (time limit) | 0.000–0.005 | 0.12–0.15 | 0.24–0.42 | 0.80–0.86 | 0.28–0.30 |

Each row used two seeds. Clipping removes nearly all the favorable-branch error, and more sampling steadily reduces the rest, though no row reaches the 0.10 gate. Root error mostly reflects the unfavorable action from comment 1.

Retrace didn't help. In one clipped seed at 10M records, its measured trace was 1.03 rounds in average-policy mode and 1.10 in best-response mode. Root errors were 0.65, 1.25 and 1.48 at 50, 100 and 200 rounds, which isn't clearly better than one-step. That supports the doc's one-step default.

- **Fix:** bound Q with clipped targets or a tanh head, since utilities are already bounded. Raise Q's sampling to at least one transition per new record, and test a replay ratio of about 4. Update the device-hour budgets to match, because the 9×9 Q cost will rise. Fix comments 1 and 2 before tuning further, since they account for much of the remaining failure.

### Medium

**4. The total cost is never stated, and NFSP's A2 work is spent even if PPO passes (§6).** The doc gives each piece but no total. On the single profiling accelerator:

| Path | Device-hours | About |
|---|---:|---:|
| A2 and controls, both learners | 264 | 11 days |
| A2, then PPO base arm and gray study | 568 | 24 days |
| A2, then NFSP base arm and gray study | 1,432 | 60 days |

These exclude promotion rounds and the CPU-only stages. The selection rule says to weigh cost, but no one has been asked whether either total is acceptable.

- **Fix:** state the totals and a go/no-go point after A2. Run cheap tasks first, starting with the tabular controls and matrix tasks, so a defect is caught before the device-hours are spent.

### Checked and working

- **Greedy-label averaging:** this fix works in the toy at 15M records. The worst gap over the final five evaluations was 0.008–0.012 on the 82-action pure control, 0.023–0.029 on the asymmetric 2×2 and 0.023–0.027 on rock-paper-scissors, all under the 0.05 gate. On the pure control, 0.3–0.5% of the labels were on non-best actions.
- **Not tested:** I didn't test PPO's new entropy floor on the short tasks.

---

## Sixth review (Claude, 2026-10-31)

This reviews the revision that adds coverage episodes, separate grading for delayed credit, bounded Q, the higher Q replay ratio and the PPO-first staged route, and reports medium or higher issues only. I found three: two high and one medium. The fifth-review fixes themselves work in my toy runs. The remaining problems are about where the route will actually lead and whether its budget allows the tests to finish.

These are the same compact toy learners as the last review, now at the revised settings, with three seeds per row. Real trainers may differ, so treat this as evidence, not proof.

### High

**1. PPO will probably fail the cheap short tasks, so the expected route and cost are NFSP's, not PPO's (§4 Phase A2, §6.1, §6.4).** PPO goes first, and its 0.001 entropy floor is described as an anti-collapse trial. In the toy, the floor didn't prevent collapse. Raising entropy only trades collapse for regularization bias, and no setting passes both tasks. The table shows the worst exact gap over the final five evaluations, against a gate of 0.05:

| Entropy weight | Latest, asymmetric 2×2 | Latest, rock-paper-scissors | Mixture, asymmetric 2×2 | Mixture, rock-paper-scissors |
|---|---:|---:|---:|---:|
| Doc's taper to 0.001 | 0.85–1.15 | 1.00 | 0.15–1.15 | 0.37–1.00 |
| Fixed 0.01 | 0.85–1.15 | 1.00 | 0.21–0.85 | 0.25–0.76 |
| Fixed 0.03 | 0.84–1.08 | 0.98–1.00 | 0.06–0.22 | 0.20–0.32 |
| Fixed 0.1 | 0.27–0.40 | 0.18–0.39 | 0.02–0.08 | 0.07–0.11 |
| Fixed 0.3 | 0.10–0.11 | 0.10–0.12 | 0.07–0.08 | 0.02–0.03 |

The first row covers runs with and without the KL stop; the fixed rows ran without it. The cheap-first order is the right design, because it would find this within hours. But §6.4 presents 364 device-hours as the PPO path and leaves `H_N` open, so a go/no-go decision could be made around a total that probably won't apply. The NFSP total is `352 + 12*H_N`, and `H_N` will rise if the study needs `R=4`.

- **Fix:** run the cheap PPO matrix tasks with the real trainer before anyone relies on the PPO total. If they fail, plan around NFSP's cost from then on, and profile `H_N` at the selected replay ratio before the A2 exit decision. Add one higher-entropy setting to the trial, so a failure isn't only a verdict on 0.001.

**2. NFSP's budget ends before its delayed-credit test can pass (§4 Phase B2, §6.2).** `R=1` is the default and `R=4` is a conditional comparison capped at 5M records per seed. Neither setting passes at its allowed budget, and the setting that gets close is cut off too early:

| One-step Q, worst early-state error | Records | 50 rounds | 100 rounds | 200 rounds |
|---|---:|---:|---:|---:|
| `R=1` | 5M | 0.06–0.18 | 0.36–0.59 | 1.99 |
| `R=1` | 14M | 0.005–0.009 | 0.016–0.039 | 0.13–0.81 |
| `R=4` | 5M (the doc's cap) | 0.009–0.014 | 0.033–0.042 | 1.97–1.98 |
| `R=4` | 10M (my time limit) | 0.001–0.033 | 0.004–0.064 | 0.02–0.17 |

The gate is 0.10. `R=1` is still improving but misses at 200 rounds near its 15M cap. `R=4` is close by 10M, though one seed finished at 0.17. At 5M, all three seeds are still far from passing at 200 rounds, so the conditional `R=4` comparison can't show the benefit it exists to measure.

The current order also spends 144 device-hours on a one-step/Retrace comparison at `R=1`, which is likely to fail for both candidates and leaves the `R` question unanswered. I didn't run Retrace at the new settings, because my toy was too slow at 512 sequence starts. Its measured trace of about 1.1 rounds in the last review gives little reason to expect a different result.

- **Fix:** give the `R=4` comparison the same 15M cap, and budget it from the profile. Settle `R` first using one-step only. Run Retrace afterward, and only if one-step still fails at the chosen `R`, which saves most of the 144 hours if it doesn't.

### Medium

**3. A2 no longer tests the deployed policy on long delayed credit, and the first real check is expensive (§4 Phase A2, §6.3).** Moving this family's grade to greedy Q was the right way to separate delayed credit from averaging. But the average policy is still what ships, and in these runs it lags well behind Q:

| One-step Q, 200 rounds | Greedy root correct from | Greedy gap | Average-policy gap |
|---|---:|---:|---:|
| `R=1`, at 14M | 2M–7M, varying by seed | 0.0 | 0.19–0.85 |
| `R=4`, at 10M | 4M–5M | 0.0 | 0.45–0.58 |

Long-horizon decisions at 9×9 are the early-game moves, so the same lag could weaken the deployed policy there. The first place that gets graded is NFSP screening at 20M, after substantial spending, and a failure then would be hard to interpret.

Burn-in isn't a free fix either. Clearing the reservoir halfway through the run should remove the chain lag, since Q is correct by then, but it hurts the short mixed tasks, which then have less time to average:

| 15M records, worst gap over final five evaluations | Whole-run reservoir | Reservoir cleared at half |
|---|---:|---:|
| Asymmetric 2×2 | 0.023–0.029 | 0.026–0.094 |
| Rock-paper-scissors | 0.023–0.027 | 0.045–0.051 |
| 82-action pure control | 0.008–0.012 | 0.000 |

- **Fix:** move the burn-in question out of the optional 9×9 experiment and into A2, where it costs little. Try an earlier opening point on the short tasks and the chain, and record a decision before NFSP's 9×9 pilot. Set a threshold for the average-policy gap on the chain, even if it's only a warning, so that the lag is a decision rather than a reported number.

### Checked and working

- **Coverage episodes:** PPO's unfavorable-branch error at 200 rounds fell from 0.28–0.76 to at most 0.001 at 10M records, with a policy gap of at most 0.001. PPO now passes the delayed-credit family in all three seeds.
- **Bounded Q:** favorable-branch error stays at 0.001 or below at the final evaluations, without the earlier errors above 1.0. Unfavorable-branch error now follows the same downward trend as root error.
- **Separate grading:** the greedy root is correct at the final evaluation in all six runs, so the delayed-credit gate now measures what it's meant to.
- **Staged route and ledger:** the cheap-first order is sound, and the device-hour totals in §6.2 and §6.4 add up.

---

## Seventh review (Claude, 2026-10-31)

This reviews the revision that adds the real-trainer PPO preflight, equal `R=4` caps, the 5M reservoir opening and the readiness bands, and reports medium or higher issues only. I found no high issues and two medium ones. The design is now ready for the real work, and I recommend stopping design review here and starting Phase A: the remaining unknowns can only be answered by the real trainer, not by more toy runs.

These toy runs use the same compact learners as the last two reviews, with three seeds each. They tested the revised protocol prospectively, not by filtering old runs. Treat them as evidence, not proof.

### Medium

**1. The fixed 5M opening has little margin on the chain and no direct evidence at 9×9 (§4 Phase A2, §6.3).** The opening works very well in the toy, but its success depends on Q becoming correct just before 5M records. Average-policy gap at 200 rounds, Q error and the point where the greedy root last went wrong were:

| 5M opening, 200-round chain | Greedy root last wrong at | Average-policy gap, 14M | Worst Q error, 14M |
|---|---:|---:|---:|
| `R=1`, seed 0 | 4M | 0.002 | 0.27 |
| `R=1`, seed 1 | 7M | 0.54 | 0.31 |
| `R=1`, seed 2 | 7M | 0.56 | 0.08 |
| `R=4`, seed 0 | 5M | 0.050 | 0.012 |
| `R=4`, seed 1 | 5M | 0.084 | 0.071 |
| `R=4`, seed 2 | 4M | 0.008 | 0.005 |

These evaluations were 1M records apart, so the greedy-root times are approximate.

The cost of lateness is steep. If Q first turns correct at time `T`, the labels from 5M to `T` are wrong, and the gap is about `(T - 5M) / 10M` at the end of a 15M run. So correctness at 6M lands in the warning band and 7.5M in no-go. A real trainer that's somewhat slower than this toy could fail for that reason alone, and the plan rules out trying another opening point. `R=4` seed 1 also shows how thin the margin is: its Q error was 0.128, 0.122, 0.322, 0.134 and 0.071 over the last five evaluations, and its average-policy gap was 0.109 at 12M and 0.095 at 13M. So it would probably miss the strict final-five rule and sit on the edge of the 0.10 band, even though it ends the run passing.

At 9×9, nothing yet supports the same absolute 5M. The chain has no opponent, and Q is correct by then. In real play, average mode is uniform for those 5M records, so nearly all games are against a near-random opponent. Uniform games often run close to the 200-round limit, so that's only about 15,000 games. Q at opening will mostly know how to exploit random play, and the reservoir's first labels come from that. It isn't obvious that 5M buys any readiness there, and it's a quarter of the 20M screening budget.

- **Fix:** before running, predeclare one later fallback opening point for the chain and short tasks, along with what a marginal miss on one seed means, so that a near-pass doesn't turn into an improvised decision. In the 9×9 pilot, measure greedy Q's strength against random and tactical opponents at the opening point, and record a go/no-go on keeping 5M. If Q is still weak or only exploits random play, test a pre-opening opponent that uses Q itself with exploration, instead of uniform play.

**2. Nothing limits how much replay state is kept, and the actual machine has limited disk (§7, §6).** §7 lists replay contents and reservoir targets among what to save, and it says optimizer state and the archive live alongside checkpoints, but it sets no retention rule. The planned machine has about 123 GB free. I estimate a replay and reservoir snapshot at 150 MB to 1 GB. Saved at every 1M-record milestone, that's 12–80 GB per 80M run, or about 144–960 GB over the twelve NFSP runs. A full disk can kill a long run or corrupt its newest checkpoint.

- **Fix:** keep network checkpoints permanently, because the archive and evaluations need them. Keep only the latest two replay and reservoir snapshots per active run, and check free space before each run and at each checkpoint, with a stop threshold of about 20 GB. Record the actual machine in the run manifest: home-ai, with a Ryzen 9 5900X (12 cores), an RTX 3080 Ti with 12 GB, and 31 GiB of memory.

### Checked and working

- **5M opening on short tasks:** the worst gap over the final five evaluations at 15M records was 0.022–0.038 on the asymmetric 2×2, 0.027–0.037 on rock-paper-scissors and 0.000 on the 82-action pure control. All pass the 0.05 gate, and all are better than the half-run reset.
- **5M opening on the chain:** with `R=4`, this moves the long-chain average-policy gap from 0.45–0.58 under whole-run averaging to 0.008–0.084, which reaches the eligible band at the last evaluation. `R=4` is clearly the setting to expect, because `R=1` still has Q errors of 0.08–0.31 at 14M and ends in no-go on the average policy in two of three seeds.
- **PPO preflight protocols:** neither passes in the toy, so the route's fallback to NFSP is well placed. The fixed 0.1 setting with the KL stop gave latest-policy gaps of 0.18–0.40 and mixture gaps of 0.02–0.08 and 0.07–0.11. These match the earlier run without the stop, which suggests the stop never triggered at that entropy. The real trainer still has to confirm this.
- **Hardware:** I estimate home-ai can meet the doc's ceilings, at about 35–40 hours per 80M-record NFSP run at `R=4`, or about 20 days for the twelve runs. That's a rough estimate of `H_N` until the profiling step replaces it.

## Eighth review (Claude, 2026-11-01)

This reviews the revision that adds the 7M fallback, the marginal-window rule, the 9×9 opening checkpoint, the Q-driven variant and the retention rules. I found one high issue and two medium ones. Phase A can still start now, because none of them touch the resolver work. All three are small edits to the A2 and pilot plan, and they should go in before A2 runs.

This round includes the first 9×9 evidence: a throwaway one-step Q trainer on the real rules with §3's 64-channel, eight-block network at `R=4`, run on home-ai. It trains Q only, with no average-policy network, so it says nothing about the deployed policy. The opponents are uniform-random and a simple tactical agent I wrote, which captures, saves groups in atari, and otherwise avoids filling its own eyes or self-atari. Beating that agent doesn't show strength against good play. The chain and short-task results come from the same compact toy learners as earlier reviews. Treat all of this as evidence, not proof.

### High

**1. As written, the route is likely to stop, and a one-parameter change plus an existing variant avoids that (§4 B2, §6.3).** The default uniform pre-opening fails the 9×9 opening checkpoint. The only behavior alternative, Q-driven pre-opening, then fails its required chain recheck at the current target refresh. §6.3 says to stop the route if neither alternative passes. Combining Q-driven pre-opening with a 12,800-record refresh was the only one of these four combinations that passed both tests:

| `R=4`, 5M opening | Greedy root last wrong | Chain worst Q error, final five | Chain worst average gap, final five | 9×9 greedy Q vs tactical at 5M |
|---|---:|---:|---:|---:|
| Uniform, refresh 25,600 (the doc's default), 5 seeds | 4–5M | 0.007–0.168 (2 of 5 above 0.10) | 0.009–0.095 | -1.00, -1.00, -1.00 |
| Uniform, refresh 12,800, 5 seeds | 1–2.5M | 0.002–0.007 | 0.000 | +0.19, -0.94 (2 seeds) |
| Q-driven, refresh 25,600, 5 seeds | 4.5–5M | 0.074–0.964 (4 of 5 above 0.10) | 0.008–0.055 | +1.00, +0.98, +1.00 |
| Q-driven, refresh 12,800, 3 seeds | 1.5–2.5M | 0.002 | 0.000 | +0.95, +1.00, +1.00 |

- **Uniform pre-opening:** this confirms the concern in my seventh review. At 5M, Q scores +0.49 to +0.89 against uniform-random but -1.00 against the tactical agent, so it has mostly learned to exploit random play. Waiting doesn't help: at 7M it still scores -0.65, -0.95 and -0.42. From 3M to 8M, none of its 63 evaluations clears the checkpoint's bound, using about 2.5 standard errors at 500 games.
- **Q-driven pre-opening:** 62 of 63 evaluations clear the same bound from 3M to 8M at refresh 25,600, and 39 of 39 from 3M to 6M at refresh 12,800. On short tasks it passes at the 5M opening: 0.019–0.031 on the asymmetric 2×2, 0.033–0.050 on rock-paper-scissors and 0.000 on the 82-action control. One rock-paper-scissors seed sits right at the limit.
- **Why refresh matters:** with one-step targets, value moves about one step per target refresh on the opaque chain. 200 rounds × 25,600 is about 5.1M records, and 200 × 12,800 is about 2.6M, which match the measured times. So the thin margin I reported last time comes from the refresh interval, not from chance. At 25,600, the 5M opening falls almost exactly on the point where Q turns correct.
- **The marginal rule doesn't rescue the default:** the two uniform seeds above 0.10 reach 0.134 and 0.168, beyond the 0.12 marginal limit, so neither qualifies for the extension.
- **`R=4` stays necessary:** `R=1` at refresh 12,800 gets the average gap to 0.010 or less, but two of three seeds still have Q errors above 1.3.
- **Fix:** before A2, change the target refresh to 12,800 fresh Q-eligible records for the delayed family and 9×9, and make Q-driven pre-opening the primary protocol. Keep uniform as the comparison arm. Refresh stays fixed in fresh-record units, so the reuse comparison is still unconfounded. The real trainer must still pass the bounded Gray Go subgames and the meaningful-branch control, which none of these runs tested. My one-step matrix tasks have no target network, so they say nothing about refresh.

### Medium

**2. The 7M fallback fails as written, and it's the wrong lever (§4 A2, §6.2).** At its 15M cap, the worst gap over the final five evaluations was 0.074–0.086 on the asymmetric 2×2 and 0.044–0.072 on rock-paper-scissors, against a 0.05 limit. Most of these runs would miss the gate, and the rule says to reject the fallback when that happens. At 17M, which gives it the same 10M of post-opening records as the 5M protocol, all six runs pass at 0.021–0.043. Even then, on the chain, its average gap is 0.000 in all three seeds, but seed 1's Q error stays at 0.59–1.25 from 13M to 15M and is still 0.39 at 17M. At 9×9, Q under uniform pre-opening is still losing to the tactical agent at 7M.

- **Fix:** remove the 7M fallback in favor of the refresh change. If you keep it, cap it at 17M and update `CS7` in the ledger. The seventh review's suggestion of a later opening point is what led here, and these results show it doesn't address the cause.

**3. The opening checkpoint reads one frozen Q, and Q swings too much for that (§6.3).** The result can be wrong in both directions. In the uniform refresh-12,800 arm, seed 0 scored -0.77, -1.00 and then +0.19 at exactly 5M, the only clear pass among its arm's 19 evaluations from 3M on, so a single read would call it ready. In the Q-driven refresh-25,600 arm, seed 2 scored +1.00 at 5M, -0.20 at 5.25M and +1.00 at 5.5M, so a checkpoint at 5.25M would have called it not ready.

- **Fix:** evaluate the last three checkpoints, 250k records apart, and require at least two to clear the bound. Applied to these runs, that rejects the uniform case and accepts the Q-driven one. Keep the doc's confidence procedure for each read.

### What this changes in my earlier reviews

- **Seventh review, 9×9 concern:** this held up. It was only reasoning then, and it's now measured against this tactical agent.
- **Seventh review, proposed fix:** the Q-driven variant was the right suggestion, but the later fallback wasn't. See item 2.
- **Seventh review, hardware estimate:** my 35–40 hours per 80M-record run was too high. With eight or nine runs sharing the RTX 3080 Ti, each reached 8M records in about 6.9 hours, or roughly 10 device-hours per 80M-record run and 5 days for twelve. This covers one-step Q at `R=4` and frequent evaluation, but not the average-policy network or Retrace, so it still doesn't replace the `H_N` profile.
- **Not affected:** the retention rules, the PPO findings and the recommendation to start Phase A.

### Limits

- **Q only:** nothing here shows that the average policy trained from these labels plays well at 9×9. That remains the pilot's job.
- **Strength is relative:** a Q that beats my tactical agent almost every time could still lose badly to anything stronger.
- **Short self-play games:** under Q-driven pre-opening, one seed's self-play games averaged only about 30 rounds at 5M, against about 135 for the other two at refresh 25,600. This didn't hurt it here, but the pilot should log game length so it can detect early mutual passing.
- **Few seeds:** there are three seeds at most per 9×9 arm, and the two uniform refresh-12,800 runs stopped soon after 5M. Evaluations were 250k records apart at 9×9 and 500k on the chain.

The 9×9 engine and trainer, the chain and short-task scripts, and all logs are in `~/graygo-review` on home-ai.

## Ninth review (Claude, 2026-11-01)

This is a design review only, with no new experiments, so the issues below come from reasoning about the text, not from measurement. I found one high issue and three medium ones.

**Is it ready to implement?** Phase A is ready now: the rules in §1 are precise enough to build the resolver and its tests, and I found no stale 25,600, 7M or `CS7` references in the live text. The PPO preflight and A2 are ready too, once the hand-off in item 1 is written down, because A2 must test the same rule the pilot will use. The NFSP pilot should wait for items 1–3. All four are short edits, and they don't need another review round.

### High

**1. The hand-off at opening undoes the benefit of Q-driven pre-opening (§4 A2, §4 B2, §6.3).** Average mode switches from epsilon-greedy Q to the learned average policy at the *first supervised update*. That update needs only one minibatch of 512 labels, or about 5,000 records after opening, because only about 10% of records earn labels. So 90% of episodes then sample from a network trained on a single batch, which plays close to arbitrarily.

This recreates the condition the eighth review measured as harmful: Q trained mostly against a very weak opponent. It also happens at the worst time. The 500,000-record replay turns over completely by about 5.5M, so Q can lose its strength against the tactical agent while it's supplying the reservoir's first labels, and those labels are never removed. The three-checkpoint decision can't catch this, because it reads Q before opening. After that, the doc only logs results, so a collapse would not stop the run.

My eighth-review runs don't test this. They stopped short of any hand-off, and they had no average-policy network. The toy short tasks and chain passed with an immediate switch, but they have no long game for Q to forget. I recommended Q-driven pre-opening without specifying this step, so the gap traces back to that recommendation.

- **Fix:** replace "first supervised update" with a fixed ramp. From 5M to 7M records, raise linearly the chance that an average-mode episode samples the average policy rather than epsilon-greedy Q, drawn once per episode. Log which policy acted. A fixed schedule is simple and can't be tuned after seeing results. Use the same rule in A2.
- **Pilot change:** extend the pilot to 8M so it covers the whole ramp. Repeat the Q check against both opponents at 6M, 7M and 8M with the existing two-of-three rule, and treat a miss as no-go for screening. §6.3's record and time limits need updating to match.

### Medium

**2. The pilot's go/no-go has no seed count or rule across seeds (§6.3).** The table gives no seed count for the pilot, and "no-go for that configuration" doesn't say how seeds combine. The confirmation stage has it worse: the study's twelve runs would each reach an opening point, and if one misses, dropping it would bias the study toward lucky seeds.

- **Fix:** run the pilot with three seeds and require all three to pass. In screening and confirmation runs, always open at the fixed point, report the checkpoint result, and don't drop a seed because of it.

**3. NFSP screening has no thresholds, and the gate's opponent isn't pinned down (§6.3, §4 Phase A).** "Random/tactical thresholds" at 20M has no numbers, yet it decides whether to fund the confirmation runs. Every 9×9 gate also depends on a tactical agent the doc describes only as simple and documented. If that agent breaks ties deterministically, a greedy Q playing it can repeat the same few games, and the confidence bound would no longer be valid.

- **Fix:** use PPO's bars for the deployed average policy: a positive lower 95% bound against random and at least `+0.10` against tactical. Specify the tactical agent's move priorities in Phase A, require random tie-breaking or randomized openings, check that evaluation games aren't duplicates, and freeze the agent before the pilot.

**4. The 9×9 epsilon schedule has no defined length (§4 B2).** It runs "over first half of budget", but a 9×9 run passes through 6M or 8M, 20M and 80M stages. Read against the pilot, epsilon would reach `0.01` by about 3M or 4M. Read against confirmation, it would still be about `0.09` at 5M, which is what my eighth-review runs used. Under Q-driven pre-opening, epsilon is the main source of variety in play before opening, so a pilot on the short reading wouldn't predict the longer runs.

- **Fix:** fix the 9×9 schedule at `0.1 -> 0.01` over the first 40M records for every stage, so the pilot and screening are exact prefixes of a confirmation run.

### Checked and fine

- **Rules and encoding:** the removal order, ko and clearing tables, scoring totals and input channel counts are consistent.
- **Refresh and ledger:** the 25 and 100 Q updates per refresh are right, and `N_A2` no longer carries retired terms.
- **Opening checkpoint:** the fixed window and its rule against mixing passes across snapshots are sound for what they read.
- **Retention:** the rules cover the closed-reservoir state, because it stays among the latest bundles when a pilot stops there.
