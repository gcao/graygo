from copy import deepcopy
from decimal import Decimal
import unittest
from contextlib import redirect_stderr
import io
import tempfile
from pathlib import Path
from graygo.task_training import NFSPTaskConfig
from tools.nfsp_delayed_cpu_prices import profile_price,combine_source_prices,check_case,VERSION,main,digest,inputs_match


class DelayedCPUPriceTests(unittest.TestCase):
    def fixture(self,reuse=4):
        configs={p:NFSPTaskConfig(time_cap_seconds=300,records=65536,eval_every=32768,q_reuse=reuse,purpose='profile',reservoir_open=5000000 if p=='pre' else 0,handoff_end=7000000 if p=='pre' else 1000).plan() for p in ('pre','post')}
        rows=[{'phase':p,'records':65536,'status':'complete','learner':'neural','q_reuse':reuse,'formal_protocol':False,'seconds_including_setup':40 if p=='pre' else 45,'reservoir_size':0 if p=='pre' else 8000,'counters':{'fresh_records':65536,'q_updates':480,'target_refreshes':5,'supervised_updates':0 if p=='pre' else 100},'acting_policy_counts':{'average':0 if p=='pre' else 58000}} for p in ('pre','post')]
        return rows,configs

    def test_twofold_cpu_price_preserves_complete_record_scope_and_no_accelerator_cost(self):
        rows,configs=self.fixture();price=profile_price(4,rows,configs);expected=(Decimal(5000000)*40+Decimal(10000000)*45)/65536
        self.assertEqual(Decimal(price['phase_weighted_15M_seconds']),expected);self.assertEqual(price['device_hours'],'0');self.assertGreaterEqual(Decimal(price['cpu_process_ceiling_seconds']),2*expected);self.assertEqual(price['formal_record_cap'],15000000);self.assertEqual(price['original_seed_scope'],[0,1,2])

    def test_missing_duplicate_incomplete_or_no_average_work_cannot_price_a_learner(self):
        rows,configs=self.fixture()
        variants=[rows[:1],[rows[0],rows[0]]]
        for field,value in [('status','time_limited'),('acting_policy_counts',{'average':0}),('counters',{'fresh_records':65536,'q_updates':1,'target_refreshes':1,'supervised_updates':0})]:
            bad=deepcopy(rows);bad[1][field]=value;variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):profile_price(4,bad,configs)

    def test_cuda_native_changed_schedule_seed_learner_or_target_work_is_rejected(self):
        rows,configs=self.fixture()
        for field,value in [('device','cuda'),('storage','rule-state'),('hidden',32),('seed',1),('learner','tabular'),('target_every',25600),('uniform_comparison',True),('handoff_end',2000)]:
            changed=deepcopy(configs);changed['post'][field]=value
            with self.assertRaises(ValueError,msg=field):profile_price(4,rows,changed)
        for value in [0,-1,float('nan'),True]:
            changed=deepcopy(rows);changed[0]['seconds_including_setup']=value
            with self.assertRaises(ValueError):profile_price(4,changed,configs)

    def test_direct_selected_source_is_priced_and_neither_original_nor_new_cost_is_dropped(self):
        rows,configs=self.fixture();reference=profile_price(4,rows,configs);changed=deepcopy(rows);changed[1]['seconds_including_setup']=90;selected=profile_price(4,changed,configs)
        merged=combine_source_prices(reference,selected);self.assertEqual(merged['cpu_process_ceiling_seconds'],selected['cpu_process_ceiling_seconds']);self.assertEqual(merged['selected_package_forecast_seconds'],selected['phase_weighted_15M_seconds'])
        reversed_cost=combine_source_prices(selected,reference);self.assertEqual(reversed_cost['cpu_process_ceiling_seconds'],selected['cpu_process_ceiling_seconds']);self.assertEqual(reversed_cost['selected_package_forecast_seconds'],reference['phase_weighted_15M_seconds'])
        with self.assertRaises(ValueError):combine_source_prices(reference,selected|{'q_reuse':1})
        with self.assertRaises(ValueError):combine_source_prices(reference,selected|{'device_hours':'1'})

    def test_formal_applicability_is_bound_to_package_cpu_runtime_exact_recipe_and_ceiling(self):
        rows,configs=self.fixture();price=profile_price(4,rows,configs)|{'raw_budget_sha256':'a'*64};identity={k:{} for k in ['hostname','cpu','python','platform','dependencies','runtime']}
        proof={'version':VERSION,'status':'qualified_prospective_cpu_delayed_prices','target_runtime_source_sha256':'b'*64,'prices':{'C4':price},'runtime_identity':identity}
        config=NFSPTaskConfig(time_cap_seconds=float(price['cpu_process_ceiling_seconds']),seed=2,q_reuse=4,budget_profile_sha256='a'*64).plan();manifest={'source_sha256':'b'*64,'config':config,**identity}
        result=check_case(manifest,proof,'C4');self.assertTrue(result['price_applies']);self.assertFalse(result['execution_authorized']);self.assertFalse(result['a2_complete'])
        for field,value in [('device','cuda'),('task','branching'),('seed',3),('q_reuse',1),('records',10000000),('reservoir_open',0),('time_cap_seconds',float(price['cpu_process_ceiling_seconds'])+900),('budget_profile_sha256','c'*64)]:
            changed=deepcopy(manifest);changed['config'][field]=value
            with self.assertRaises(ValueError,msg=field):check_case(changed,proof,'C4')
        changed=deepcopy(manifest);changed['source_sha256']='d'*64
        with self.assertRaises(ValueError):check_case(changed,proof,'C4')
        changed=deepcopy(manifest);changed['runtime']={'device':'cuda'}
        with self.assertRaises(ValueError):check_case(changed,proof,'C4')

    def test_serialized_snapshot_string_paths_are_read_and_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.json';path.write_text('immutable original')
            snapshot={str(path):digest(path)};self.assertEqual(digest(str(path)),digest(path));self.assertTrue(inputs_match(snapshot));self.assertFalse(inputs_match({}))
            path.write_text('changed model');self.assertFalse(inputs_match(snapshot))

    def test_cli_refuses_to_overwrite_any_existing_input_or_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'original.json';path.write_text('original evidence')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['--repo',temporary,'--proposal',str(path),'--audits',temporary,'--selected',temporary,'--output',str(path)])
            self.assertEqual(path.read_text(),'original evidence')


if __name__=='__main__':unittest.main()
