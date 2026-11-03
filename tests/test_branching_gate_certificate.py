from fractions import Fraction
import unittest
from tools.branching_gate_certificate import certify, controller_worst_value, opponent_exploiting_value


class BranchingGateCertificateTests(unittest.TestCase):
    def test_actual_branch_matrices_and_every_role_exchange_have_exact_mixed_solutions(self):
        result = certify()
        self.assertEqual([row['controller_value'] for row in result['branch_equilibria']], ['-1/7', '-7/22'])
        self.assertEqual(result['branch_equilibria'][0]['controller_action0_probability'], '3/7')
        self.assertTrue(result['all_actual_role_swapped_payoffs_verified'])

    def test_all_pure_and_uniform_tie_behavioral_strategies_fail_original_greedy_gate(self):
        result = certify()
        self.assertEqual(result['controller_behavioral_strategies_enumerated'], 27)
        self.assertEqual(result['opponent_behavioral_strategies_enumerated'], 9)
        self.assertEqual(result['controller_minimum_complete_strategy_gap'], '3/28')
        self.assertEqual(result['opponent_minimum_complete_strategy_gap'], '1/7')
        self.assertFalse(result['legacy_greedy_equilibrium_gate_feasible'])
        for role in ('controller_enumeration', 'opponent_enumeration'):
            self.assertTrue(all(Fraction(row['gap']) > Fraction(1, 20) for row in result[role]))
        self.assertFalse(result['original_data_or_grades_modified'])
        self.assertFalse(result['a2_complete'])

    def test_unrestricted_mixed_policy_meets_value_so_obstruction_is_not_the_game(self):
        matrices = (((Fraction(1), Fraction(-1)), (Fraction(-1), Fraction(1, 2))),
                    ((Fraction(1, 2), Fraction(-1)), (Fraction(-1), Fraction(1, 4))))
        value = Fraction(-1, 7)
        self.assertEqual(controller_worst_value(matrices, (Fraction(1), Fraction(3, 7), Fraction(5, 11))), value)
        self.assertEqual(opponent_exploiting_value(matrices, (Fraction(3, 7), Fraction(5, 11))), value)


if __name__ == '__main__':unittest.main()
