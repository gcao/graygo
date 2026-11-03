from copy import deepcopy
import unittest
from tools.task_domain_boundary import complete_domain,verify_pair,CASES,VERSION


class TreeGame:
    def terminal(self,state):return len(state)==2
    def actions(self,state,role):return (0,1) if not self.terminal(state) else ()
    def transition(self,state,b,w):return state+((b,w),)
    def utility(self,state):return (sum(b-w for b,w in state))/2


class DomainBoundaryTests(unittest.TestCase):
    def test_every_joint_edge_is_enumerated_with_shared_roots(self):
        graph=complete_domain(TreeGame(),[(),()],lambda s:list(s),max_nodes=30,max_edges=25)
        self.assertEqual(len(graph['states']),21)
        self.assertEqual(len(graph['edges']),20)
        self.assertEqual(len(graph['terminal_ids']),16)
        self.assertEqual(len(graph['nonterminal_ids']),5)

    def test_limits_cannot_produce_a_partial_pass(self):
        for nodes,edges in ((20,25),(30,19)):
            with self.assertRaises(ValueError):complete_domain(TreeGame(),[()],lambda s:list(s),max_nodes=nodes,max_edges=edges)

    def test_state_encoder_cannot_alias_different_public_states(self):
        with self.assertRaisesRegex(ValueError,'aliases'):
            complete_domain(TreeGame(),[()],lambda s:len(s),max_nodes=30,max_edges=25)

    def test_terminal_utility_and_empty_legality_are_rejected(self):
        class Bad(TreeGame):
            def utility(self,state):return float('nan')
        with self.assertRaises(ValueError):complete_domain(Bad(),[()],lambda s:list(s),max_nodes=30,max_edges=25)
        class Empty(TreeGame):
            def actions(self,state,role):return ()
        with self.assertRaises(ValueError):complete_domain(Empty(),[()],lambda s:list(s),max_nodes=30,max_edges=25)

    def test_changed_edge_output_seed_or_work_inventory_is_rejected(self):
        def result(source):
            return {'version':VERSION,'status':'complete','source_sha256':source,'source_unchanged':True,'new_training_records':0,'new_scored_evaluation_games':0,'execution_authorized':False,'runtime':{'device':'cpu','threads':1,'deterministic':True},
                    'cases':{f'{t}/{l}/{s}':{'complete':True,'config':{'task':t,'learner':l,'seed':s,'purpose':'profile','device':'cpu','storage':'features','q_reuse':4,'width':64,'blocks':8,'norm_groups':8},'counts':{'states':3,'nonterminal':1,'terminal':2,'joint_transitions':2,'observations':2,'raw_forwards':4},'complete_domain_sha256':'a'*64} for t,l,s in CASES}}
        a,b=result('a'*64),result('b'*64)
        passed=verify_pair(a,b,reference_source='a'*64,candidate_source='b'*64);self.assertEqual(passed['case_count'],12);self.assertFalse(passed['a2_evidence_reusable'])
        for mutate in (lambda r:r['cases'].pop('gray-pure/neural/2'),lambda r:r['cases']['gray-mixed/tabular/1'].update(complete_domain_sha256='b'*64),
                       lambda r:r['cases']['gray-pure/neural/0']['counts'].update(observations=1),
                       lambda r:r['cases']['gray-pure/neural/0']['config'].update(seed=9),lambda r:r['runtime'].update(device='cuda')):
            changed=deepcopy(b);mutate(changed)
            with self.assertRaises(ValueError):verify_pair(a,changed,reference_source='a'*64,candidate_source='b'*64)
