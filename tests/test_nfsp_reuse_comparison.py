import ast
from copy import deepcopy
import inspect
import unittest
from tests import test_a2_cohort as fixtures
from tools import a2_cohort, a2_source_cohort, nfsp_reuse_comparison as comparison


class ReuseComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.A2ProtocolTests(methodName='runTest')
        self.plan = self.fixture.plan()

    def make_case(self, primary=False):
        wrapper = self.fixture.case(self.plan, 'delayed', 'neural', 1)
        self.fixture.mutate(wrapper, lambda body: body.update(primary_recipe=primary))
        return wrapper

    def test_original_numeric_checker_is_preserved_except_scoped_primary_guard(self):
        expected = inspect.getsource(a2_cohort.checked_case).replace('def checked_case(', 'def checked_comparison(')
        expected = expected.replace("if not body['primary_recipe']:return key,body,{'status':'diagnostic_only','numeric_passed':False}",
                                    "require(key[0]=='delayed' and key[1]=='neural','only delayed neural comparison cases are in scope')")
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(comparison.checked_comparison))))

    def test_comparison_scoring_does_not_mutate_primary_flags_or_fill_source_aware_slots(self):
        wrapper = self.make_case(); before = deepcopy(wrapper)
        key, body, score = comparison.checked_comparison(self.plan, wrapper)
        self.assertEqual(key, ('delayed', 'neural', 1))
        self.assertTrue(score['q_credit_passed']); self.assertFalse(body['primary_recipe'])
        self.assertEqual(wrapper, before)
        self.assertEqual(a2_cohort.checked_case(self.plan, wrapper)[2]['status'], 'diagnostic_only')
        with self.assertRaises(ValueError):
            a2_source_cohort.assess_sources(self.plan, {self.plan.sha256: self.plan}, {'a' * 64: {'delayed'}}, [wrapper])

    def test_original_primary_scores_and_nonprimary_numeric_scores_agree(self):
        primary = self.make_case(primary=True); other = deepcopy(primary)
        self.fixture.mutate(other, lambda body: body.update(primary_recipe=False))
        left = a2_cohort.checked_case(self.plan, primary)[2]
        right = comparison.checked_comparison(self.plan, other)[2]
        left.pop('case_sha256'); right.pop('case_sha256')
        self.assertEqual(left, right)

    def test_prior_window_value_miss_is_retained_despite_passing_final_and_average(self):
        wrapper = self.make_case()
        self.fixture.mutate(wrapper, lambda body: body['evaluations'][-2].update(max_early_error=.11698))
        self.fixture.mutate(wrapper, lambda body: body['evaluations'][-1].update(max_early_error=.09))
        score = comparison.checked_comparison(self.plan, wrapper)[2]
        self.assertFalse(score['q_credit_passed']); self.assertEqual(score['status'], 'numeric_miss')
        self.assertEqual(score['average_band'], 'at_most_0.10')
        self.assertEqual(score['violations'], [{'records': 14500000, 'metric': 'max_early_error', 'value': .11698, 'threshold': .1}])

    def test_credit_pass_and_averaging_miss_remain_separate(self):
        wrapper = self.make_case()
        self.fixture.mutate(wrapper, lambda body: [row.update(max_average_gap=.2) for row in body['evaluations'][-5:]])
        score = comparison.checked_comparison(self.plan, wrapper)[2]
        self.assertTrue(score['q_credit_passed']); self.assertFalse(score['numeric_passed'])
        self.assertEqual(score['average_band'], 'averaging_lag_0.10_to_0.25')

    def test_other_tasks_partial_records_missing_reads_and_unverified_data_cannot_pass(self):
        for family, learner in [('gray-pure', 'neural'), ('delayed', 'tabular')]:
            with self.assertRaises(ValueError):
                comparison.checked_comparison(self.plan, self.fixture.case(self.plan, family, learner, 1))
        for change in [lambda body: body.update(fresh_audit_verified=False),
                       lambda body: body['evaluations'].pop(),
                       lambda body: body.update(configured_records=5000000)]:
            wrapper = self.make_case(); self.fixture.mutate(wrapper, change)
            with self.assertRaises(ValueError):
                comparison.checked_comparison(self.plan, wrapper)
        wrapper = self.make_case()
        self.fixture.mutate(wrapper, lambda body: body.update(status='budget_limited', records=14000000))
        self.assertEqual(comparison.checked_comparison(self.plan, wrapper)[2]['status'], 'budget_limited')


if __name__ == '__main__':
    unittest.main()
