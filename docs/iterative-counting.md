# Gray Go v3 — iterative territory counting

Approved November 10, 2026. V3 retains v2 full-game ko and changes scoring only.
The played stones, captures, liberties, legal moves and pass rules are unchanged.

## Mark first, then count

Begin a fresh count from the actual resolved board. Empty points have no owner.
Use the board's four orthogonal toroidal neighbors, including neighbors across
both seams. A marked empty point acts as its assigned color for later passes.

In each pass, inspect every still-unmarked empty point using a snapshot of the
labels at the start of that pass:

- Mark it Black if at least one neighbor is a Black stone or Black mark, and all
  other neighbors are empty, Black stones, or Black marks.
- Apply the symmetric rule for White.
- A neighbor of gray prevents a Black/White mark. Gray never spreads marks.
- A point touching both Black and White, or either player together with gray,
  remains unmarked. An all-empty neighborhood also remains unmarked because it
  has no color seed.

Apply all marks from the pass simultaneously. Repeat until a pass adds none.
Marks are permanent within this count, including when opposite fronts meet
after a simultaneous pass. This makes the process independent of scan order.
Start over from the actual stones for every new count; marks are not retained
between turns.

For example, along a corridor whose other neighbors do not block propagation:

```text
B . . . G
B b . . G
B b b . G
```

The first two empty points count entirely for Black. The last remains unmarked
because it touches a Black mark and gray.

## Fractional remainder

Original Black and White stones count one point for their player. Original gray
stones count half a point each. Each Black/White mark counts one point for its
assigned player.

For each connected region of **remaining unmarked empty points**, apply the
previous faction-sharing rule. Determine its boundary colors using both actual
stones and the completed Black/White marks:

| Boundary colors | Black share per point | White share per point |
| --- | ---: | ---: |
| Black only | 1 | 0 |
| White only | 0 | 1 |
| Black + gray | 3/4 | 1/4 |
| White + gray | 1/4 | 3/4 |
| Black + White, with or without gray | 1/2 | 1/2 |
| Gray only, or no colors | 1/2 | 1/2 |

Fractional allocations do not become marks and do not propagate. The remainder
is evaluated only after marking finishes. Every intersection still contributes
one point in total, so Black + White equals N². Scores remain exact quarter points.

For this fragment, the first two empty points become Black marks:

```text
B B B B B
B . . . G
B B B B B
```

Those three empties contribute **2.75 Black / 0.25 White**, instead of the previous
2.25 / 0.75 whole-region split. The gray stone itself still contributes 0.5 each.

Marks are scoring labels only. Never place them on the live board, remove stones
because of them, change liberties, or add scoring views to ko history. All
remaining actual stones are counted as alive.

## Versions and deployment

- Arena: `graygo-v3`; Count and final results use the same iterative scorer.
- Training library: `Rules(ko="full-history", scoring="iterative")`, rule version
  `graygo/3`. State serialization and replay fingerprints bind this rule.
- `graygo-v1` and `graygo-v2` retain the original whole-region counting.
- Existing games retain their persisted ruleset. The new-game Rules selector
  exposes v3 and keeps explicit older choices available.
- Neural observations and legal move generation are unchanged by scoring.
  Existing weights can choose moves in v3, but their strength under v3 has not
  thereby been established.
- The running pinned v2 campaign and its gate evidence remain v2. Training v3
  requires a separate campaign and fresh v3 reward/evaluation data; never resume
  v2 replay as if it used the new reward.

Shared fixtures cover propagation, mixed fronts, gray boundaries and empty
boards. Regression checks cover area conservation, quarter points, color and
toroidal symmetry, score-only marks, versioned replay separation, terminal rewards,
and agreement between the Count endpoint and persisted final results.
