from copy import deepcopy
from decimal import Decimal
import unittest
from graygo.task_training import NFSPTaskConfig
from tools.nfsp_control_budget import price_pair,approved_control_limit,usage_with_approved_control_scope,VERSION
from tools.nfsp_resources import Usage,canonical,measured_usage


class ControlScopeTests(unittest.TestCase):
    def profiles(self,task='gray-mixed',learner='tabular',reuse=4):
        configs={p:NFSPTaskConfig(time_cap_seconds=130,task=task,learner=learner,q_reuse=reuse,purpose='profile',records=32768,eval_every=16384,reservoir_open=5000000 if p=='pre' else 0,handoff_end=7000000 if p=='pre' else 1000).plan() for p in ('pre','post')}
        rows=[{'phase':p,'records':32768,'status':'complete','learner':learner,'q_reuse':reuse,'formal_protocol':False,'seconds_including_setup':20 if p=='pre' else 30,'reservoir_size':0 if p=='pre' else 3000,'counters':{'fresh_records':32768,'q_updates':228,'target_refreshes':2,'supervised_updates':0 if p=='pre' else 50},'acting_policy_counts':{'average':0 if p=='pre' else 20000}} for p in ('pre','post')]
        return rows,configs

    def test_prices_keep_actual_family_learner_reuse_and_twofold_cpu_margin(self):
        for task in ('gray-pure','gray-mixed','branching','delayed'):
            for learner in ('neural','tabular'):
                rows,configs=self.profiles(task,learner);price=price_pair(task,learner,4,rows,configs);forecast=Decimal(400000000)/32768
                self.assertEqual(Decimal(price['phase_weighted_15M_seconds']),forecast);self.assertEqual(price['task'],task);self.assertEqual(price['learner'],learner);self.assertEqual(price['device_hours'],'0');self.assertGreaterEqual(Decimal(price['cpu_process_ceiling_seconds']),2*forecast)
        rows,configs=self.profiles('delayed','tabular',1);self.assertEqual(price_pair('delayed','tabular',1,rows,configs)['q_reuse'],1)

    def test_missing_work_gpu_changed_recipe_or_comparison_alias_cannot_get_a_price(self):
        rows,configs=self.profiles()
        for field,value in [('device','cuda'),('learner','neural'),('seed',1),('capacity',200000),('target_every',25600),('reservoir_open',1000)]:
            changed=deepcopy(configs);changed['pre'][field]=value
            with self.assertRaises(ValueError,msg=field):price_pair('gray-mixed','tabular',4,rows,changed)
        with self.assertRaises(ValueError):price_pair('gray-mixed','tabular',4,rows[:1],configs)
        changed=deepcopy(rows);changed[1]['acting_policy_counts']['average']=0
        with self.assertRaises(ValueError):price_pair('gray-mixed','tabular',4,changed,configs)
        with self.assertRaises(ValueError):price_pair('branching','tabular',1,rows,configs)

    def scope(self):
        proposal={'version':VERSION,'selected':False,'proposed_control_ceiling_seconds':'259200','cases':[{'id':'first'},{'id':'second'}]}
        decision={'approved':True,'proposal_sha256':canonical(proposal),'control_ceiling_seconds':'259200','case_ids':['first','second']};return proposal,decision

    def test_larger_control_limit_requires_explicit_matching_owner_decision(self):
        proposal,decision=self.scope();self.assertEqual(approved_control_limit(proposal,decision),Decimal(259200))
        for changed in [None,decision|{'approved':False},decision|{'proposal_sha256':'0'*64},decision|{'control_ceiling_seconds':'300000'},decision|{'case_ids':['first']}]:
            with self.assertRaises(ValueError):approved_control_limit(proposal,changed)

    def test_explicit_scope_review_preserves_all_costs_units_and_original_default(self):
        proposal,decision=self.scope();row=Usage('a'*64,'control','cpu_process_seconds',Decimal(50000),('b'*64,),'c'*64).to_dict();inventory={'version':'graygo/nfsp-resource-ledger/1','measurements':[row]};before=deepcopy(inventory)
        original=measured_usage([Usage(**row)]);self.assertFalse(original['cpu_categories']['control']['within_ceiling'])
        revised=usage_with_approved_control_scope(inventory,proposal,decision);self.assertTrue(revised['cpu_categories']['control']['within_ceiling']);self.assertEqual(revised['cpu_categories']['control']['actual_process_seconds'],'50000');self.assertEqual(revised['control_scope_decision']['original_default_ceiling_seconds'],'43200');self.assertEqual(measured_usage([Usage(**row)])['cpu_categories']['control']['ceiling_seconds'],'43200');self.assertEqual(inventory,before);self.assertFalse(proposal['selected'])


if __name__=='__main__':unittest.main()
