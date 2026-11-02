from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import unittest
import numpy as np
from graygo.engine import BLACK,WHITE,Rules,State,resolve,transform
from graygo.confidence import canonical_sha
from graygo.probe_certification import collision_range,CollisionRange,CertificateBudget,CertifiedRoot,certify_one_round,grade_root,qualify_pair,geometric_root_id
from graygo.solving import FiniteSolver,markov_best_response,SolveBudgetExceeded


class ProbeCertificateTests(unittest.TestCase):
    def root(self):
        state=State(Rules(3,6))
        for b,w in ((1,7),(6,4),(2,5),(9,3),(9,9)):state=resolve(state,b,w).state
        return state

    def test_exact_near_equilibrium_ranges_cover_all_vertex_pairs(self):
        c=collision_range(((1,-1),(-1,1)),(0,1),(0,1),2)
        self.assertEqual(c.value,0);self.assertEqual(c.lower,F(399,800));self.assertEqual(c.upper,F(401,800))
        self.assertEqual(c.black_vertices,((F(19,40),F(21,40)),(F(21,40),F(19,40))))
        self.assertEqual(c.white_vertices,c.black_vertices)
        exact=collision_range(((1,-1),(-1,1)),(0,1),(0,1),2,gap=0)
        self.assertEqual(exact.lower,F(1,2));self.assertEqual(exact.upper,F(1,2))
        self.assertEqual(len(exact.black_vertices),1)

    def test_nonuniform_rational_value_and_distinct_player_gap_constraints(self):
        c=collision_range(((1,-1),(-1,F(1,2))),(0,1),(0,1),2)
        self.assertEqual(c.value,F(-1,7));self.assertEqual(c.equilibrium,((F(3,7),F(4,7)),)*2)
        self.assertEqual(c.black_vertices,((F(113,280),F(167,280)),(F(97,210),F(113,210))))
        self.assertEqual(c.white_vertices,((F(83,210),F(127,210)),(F(127,280),F(153,280))))
        p,q=F(97,210),F(127,280);self.assertEqual(c.lower,p*q+(1-p)*(1-q))
        p,q=F(113,280),F(83,210);self.assertEqual(c.upper,p*q+(1-p)*(1-q))

    def test_pass_is_excluded_and_action_intersections_can_be_empty(self):
        c=collision_range(((1,-1),(-1,1)),(0,1),(0,1),1)
        self.assertEqual((c.lower,c.upper),(F(361,1600),F(441,1600)))
        c=collision_range(((0,0),(0,0)),(0,2),(1,2),2)
        self.assertEqual((c.lower,c.upper),(0,0))
        c=collision_range(((0,),),(2,),(2,),2)
        self.assertEqual((c.lower,c.upper),(0,0))

    def test_many_equilibria_require_the_whole_range_and_exact_witnesses(self):
        c=collision_range(((0,0),(0,0)),(0,1),(0,1),2)
        self.assertEqual((c.lower,c.upper),(0,1))
        self.assertEqual(CollisionRange.from_dict(c.to_dict()),c)
        bad=deepcopy(c.to_dict());bad['upper']='1/2'
        with self.assertRaisesRegex(ValueError,'complete exact proof'):CollisionRange.from_dict(bad)
        bad=deepcopy(c.to_dict());bad['black_vertices'].pop()
        with self.assertRaisesRegex(ValueError,'complete exact proof'):CollisionRange.from_dict(bad)

    def test_certification_budget_exhaustion_never_returns_partial_bounds(self):
        for budget in (CertificateBudget(max_actions=1),CertificateBudget(max_vertex_bases=1),
                       CertificateBudget(max_vertices=1),CertificateBudget(max_vertex_pairs=3)):
            with self.subTest(budget=budget),self.assertRaises(SolveBudgetExceeded):collision_range(((1,-1),(-1,1)),(0,1),(0,1),2,budget=budget)
        with self.assertRaises(SolveBudgetExceeded):certify_one_round(State(Rules(9,1)))
        with self.assertRaises(ValueError):certify_one_round(State(Rules(3,2)))

    def test_real_complete_root_matches_independent_finite_best_responses(self):
        root=self.root();c=certify_one_round(root);solution=FiniteSolver().solve(root)
        self.assertAlmostEqual(float(c.certificate.value),solution[0],places=12)
        self.assertEqual(c.certificate.black_actions,solution[1]);self.assertEqual(c.certificate.white_actions,solution[2])
        p=np.ones(len(solution[1]))/len(solution[1]);q=np.ones(len(solution[2]))/len(solution[2])
        graded=grade_root(c,p,q)
        lo=markov_best_response(root,lambda state:p,WHITE);hi=markov_best_response(root,lambda state:q,BLACK)
        self.assertAlmostEqual(float(graded['black_gap']),solution[0]-lo,places=12)
        self.assertAlmostEqual(float(graded['white_gap']),hi-solution[0],places=12)
        self.assertEqual(CertifiedRoot.from_dict(c.to_dict()),c)
        forged=CertifiedRoot(root,replace(c.certificate,lower=F(-1)))
        with self.assertRaises(ValueError):grade_root(forged,p,q)
        for bad in ((-1,2),(.1,.1),(float('nan'),1)):
            with self.assertRaises(ValueError):grade_root(c,bad,q)

    def test_full_public_history_family_identity_is_geometric_and_color_invariant(self):
        root=self.root();key=geometric_root_id(root)
        exhaustive=min(transform(root,rotation=r,reflect=f,swap=swap,shift=(row,col)).canonical_json() for r in range(4) for f in (False,True) for swap in (False,True) for row in range(3) for col in range(3))
        self.assertEqual(key,canonical_sha({'kind':'complete-root-geometric-family','state':exhaustive}))
        for r in range(4):
            for f in (False,True):
                moved=transform(root,rotation=r,reflect=f,shift=(1,2),swap=True)
                self.assertEqual(geometric_root_id(moved),key)
                self.assertEqual(certify_one_round(moved).certificate.lower,certify_one_round(root).certificate.lower)
                self.assertEqual(certify_one_round(moved).certificate.upper,certify_one_round(root).certificate.upper)
        with self.assertRaises(ValueError):qualify_pair(certify_one_round(root),certify_one_round(replace(root,rules=Rules(3,6,'remove'))))

if __name__=='__main__':unittest.main()
