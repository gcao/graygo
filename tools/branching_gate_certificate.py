"""Exact infeasibility certificate for the legacy mixed-branch greedy gate.

No evaluator, threshold, original case, matrix or training run is changed.
"""
import argparse
from fractions import Fraction
import itertools
import hashlib
import json
from pathlib import Path
from graygo.tasks import BranchingGame
from graygo.engine import BLACK, WHITE
from graygo.manifest import source_digest
from graygo import models
import torch
from tools.nfsp_resources import canonical, require

VERSION = 'graygo/mixed-branch-greedy-gate-certificate/1'
SUPPORT_PROBABILITIES = (Fraction(0), Fraction(1, 2), Fraction(1))


def rational_matrix(matrix):
    require(matrix.shape == (2, 2), 'actual two-action branch matrix required')
    return tuple(tuple(Fraction(str(float(value))) for value in row) for row in matrix)


def equilibrium(matrix):
    (a, b), (c, d) = matrix
    denominator = a - b - c + d
    require(denominator != 0, 'nondegenerate actual branch matrix required')
    p = (d - c) / denominator; q = (d - b) / denominator
    value = (a * d - b * c) / denominator
    require(0 < p < 1 and 0 < q < 1, 'both actions must be genuinely mixed')
    require(all(p * matrix[0][j] + (1 - p) * matrix[1][j] == value for j in (0, 1)), 'row equilibrium equalities differ')
    require(all(q * matrix[i][0] + (1 - q) * matrix[i][1] == value for i in (0, 1)), 'column equilibrium equalities differ')
    return p, q, value


def controller_worst_value(matrices, strategy):
    root, p0, p1 = strategy
    branch_values = [min(p * matrix[0][j] + (1 - p) * matrix[1][j] for j in (0, 1))
                     for p, matrix in zip((p0, p1), matrices)]
    return root * branch_values[0] + (1 - root) * branch_values[1]


def opponent_exploiting_value(matrices, strategy):
    # The root controller chooses its branch and later row; its opponent may
    # condition its later action on that public branch, but not hidden policies.
    return max(q * matrix[i][0] + (1 - q) * matrix[i][1]
               for matrix, q in zip(matrices, strategy) for i in (0, 1))


def certify():
    game = BranchingGame(codebook_seed=0)
    matrices = tuple(rational_matrix(matrix) for matrix in game.payoffs)
    equilibria = tuple(equilibrium(matrix) for matrix in matrices)
    value = max(row[2] for row in equilibria)
    # Finite two-action scores have exactly three ordering cases. Exercise the
    # actual evaluator primitive for each, including its uniform-tie law.
    scores = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    law = models.greedy_distribution(scores, torch.ones_like(scores, dtype=torch.bool))
    require(torch.equal(law, torch.tensor([[1., 0.], [0., 1.], [.5, .5]])),
            'the actual greedy primitive does not have the certified support law')
    # Validate every terminal joint action for both controller-role exchanges
    # against the actual immutable game, rather than an unrelated toy matrix.
    for controller in (BLACK, WHITE):
        for branch in (0, 1):
            root = game.root(controller)
            state = game.transition(root, branch if controller == BLACK else 0,
                                     branch if controller == WHITE else 0)
            for black, white in itertools.product((0, 1), repeat=2):
                terminal = game.transition(state, black, white)
                actual = Fraction(str(game.utility(terminal))) * (1 if controller == BLACK else -1)
                own, other = (black, white) if controller == BLACK else (white, black)
                require(actual == matrices[branch][own][other], 'actual role-swapped branch payoff differs')
    controller_rows = []
    for strategy in itertools.product(SUPPORT_PROBABILITIES, repeat=3):
        worst = controller_worst_value(matrices, strategy)
        controller_rows.append((value - worst, strategy, worst))
    opponent_rows = []
    for strategy in itertools.product(SUPPORT_PROBABILITIES, repeat=2):
        best = opponent_exploiting_value(matrices, strategy)
        opponent_rows.append((best - value, strategy, best))
    controller_min = min(row[0] for row in controller_rows)
    opponent_min = min(row[0] for row in opponent_rows)
    threshold = Fraction(1, 20)
    require(controller_min > threshold and opponent_min > threshold, 'infeasibility does not hold for this actual task')
    def fraction(value):return str(value)
    return {'version': VERSION, 'source_sha256': source_digest(),
        'actual_branch_matrices': [[[fraction(x) for x in row] for row in matrix] for matrix in matrices],
        'branch_equilibria': [{'controller_action0_probability': fraction(p), 'opponent_action0_probability': fraction(q),
                               'controller_value': fraction(v)} for p, q, v in equilibria],
        'complete_game_controller_value': fraction(value),
        'greedy_support_action0_probabilities': [fraction(p) for p in SUPPORT_PROBABILITIES],
        'actual_greedy_primitive_orderings_verified': True,
        'greedy_implementation_sha256': hashlib.sha256(Path(models.__file__).read_bytes()).hexdigest(),
        'controller_behavioral_strategies_enumerated': len(controller_rows),
        'opponent_behavioral_strategies_enumerated': len(opponent_rows),
        'controller_minimum_complete_strategy_gap': fraction(controller_min),
        'opponent_minimum_complete_strategy_gap': fraction(opponent_min),
        'best_possible_worst_role_gap': fraction(max(controller_min, opponent_min)),
        'legacy_greedy_gap_threshold': fraction(threshold),
        'legacy_greedy_equilibrium_gate_feasible': False,
        'all_actual_role_swapped_payoffs_verified': True,
        'controller_enumeration': [{'gap': fraction(gap), 'strategy': [fraction(x) for x in strategy],
                                    'worst_value': fraction(worst)} for gap, strategy, worst in controller_rows],
        'opponent_enumeration': [{'gap': fraction(gap), 'strategy': [fraction(x) for x in strategy],
                                  'opponent_best_value': fraction(best)} for gap, strategy, best in opponent_rows],
        'original_data_or_grades_modified': False, 'a2_complete': False, 'execution_authorized': False,
        'scope': 'Exact rational finite certificate for the implemented genuinely mixed branching game. '
                 'A two-action argmax policy is pure or uniformly tied; enumerate every such public '
                 'behavioral strategy with independent choices at root and both later branches. '
                 'This relaxed strategy set includes every shared-network greedy policy. Even exact ties '
                 'cannot meet 0.05 equilibrium gap. NFSP deployed-average and native-Q value requirements '
                 'remain separate, and existing value/average misses remain unchanged. Any corrected '
                 'evaluation/fixture protocol needs explicit versioning; this certificate grants no pass.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    path = Path(args.output); require(not path.exists(), 'preserve original certificate output')
    result = certify(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: result[key] for key in ('legacy_greedy_equilibrium_gate_feasible',
        'controller_minimum_complete_strategy_gap', 'opponent_minimum_complete_strategy_gap', 'a2_complete')}))


if __name__ == '__main__':main()
