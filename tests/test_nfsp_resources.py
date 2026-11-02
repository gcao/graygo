from copy import deepcopy
from contextlib import redirect_stdout,redirect_stderr
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from tools.nfsp_resources import NFSPResourcePlan,ExtensionCeiling,Usage,TERMS,VERSION,canonical,measured_usage,main,sha


class RouteLedgerTests(unittest.TestCase):
    def plan(self,**kwargs):
        values=dict(P=24,U=5,C1=10,C4=20,CT=30,B_0=6,B_U=7,H_P=8,H_D=15,H_N=40,D=12)
        data={'name':'synthetic','learner_source_sha256':'a'*64,'selected_reuse':1,'selected_target':'one-step',
              'term_device_hours':values,'evidence_sha256':{k:canonical(k) for k in TERMS},'project_device_hour_ceiling':1000}
        data.update(kwargs);return NFSPResourcePlan(**data)

    def test_all_selected_terms_follow_the_exact_design_expression(self):
        p=self.plan(selected_reuse=4,selected_target='retrace',r4_selected=True,retrace_selected=True,zero_opening_selected=True,
            diagnostic_selected=True,extensions=(ExtensionCeiling('b'*64,'delayed',2,'d'*64),ExtensionCeiling('c'*64,'branching',3,'e'*64)),
            corrective_device_hours=24,promotion_rounds=2,other_unreused_device_hours=3)
        r=p.review();self.assertEqual(r['N_A2'],'309');self.assertEqual(r['full_study_device_hours'],'929')
        self.assertEqual(r['numerical_cost_status'],'within_device_ceiling');self.assertFalse(r['runtime_prices_independently_qualified']);self.assertFalse(r['execution_authorized'])
        self.assertEqual(NFSPResourcePlan.from_dict(p.to_dict()),p)

    def test_unselected_work_is_not_implicitly_funded_and_pilot_is_not_added_twice(self):
        r=self.plan().review();self.assertEqual(r['N_A2'],'126');self.assertEqual(r['full_study_device_hours'],'703')
        self.assertEqual(r['full_study_components']['three_H_D'],'45')
        self.assertNotIn('three_H_P',r['full_study_components'])
        self.assertEqual(r['a2_components']['C4'],'0');self.assertEqual(r['a2_components']['CT'],'0')
        # Selection flags record undertaken/committed comparison work, not
        # merely the eventual winning target. Failed comparisons still cost.
        prior=self.plan(r4_selected=True,retrace_selected=True).review()
        self.assertEqual(prior['selected_target'],'one-step');self.assertEqual(prior['selected_reuse'],1)
        self.assertEqual(prior['N_A2'],'276');self.assertEqual(prior['full_study_device_hours'],'853')

    def test_missing_prices_never_fall_back_to_the_ppo_total_or_zero(self):
        values={k:None for k in TERMS};values.update(P=0,U=0,C1=0,B_U=0)
        r=self.plan(term_device_hours=values).review();self.assertEqual(r['N_A2'],'96')
        self.assertIsNone(r['full_study_device_hours']);self.assertEqual(r['known_term_device_ceiling_subtotal'],'112')
        self.assertEqual(r['missing_required_terms'],['H_P','H_D','H_N']);self.assertEqual(r['numerical_cost_status'],'uncosted')
        values['C1']=None;r=self.plan(term_device_hours=values).review();self.assertIsNone(r['N_A2']);self.assertEqual(r['known_term_device_ceiling_subtotal'],'112')

    def test_even_known_full_formula_requires_the_explicit_pilot_input(self):
        p=self.plan();values=dict(p.term_device_hours);values['H_P']=None
        r=self.plan(term_device_hours=values).review();self.assertIn('H_P',r['missing_required_terms']);self.assertIsNone(r['full_study_device_hours'])
        with self.assertRaisesRegex(ValueError,'include H_P'):self.plan(term_device_hours=dict(p.term_device_hours,H_P=16))

    def test_component_and_project_ceiling_violations_preserve_actual_costs(self):
        r=self.plan(project_device_hour_ceiling=700).review();self.assertEqual(r['numerical_cost_status'],'over_device_ceiling_or_component_limit')
        values=dict(self.plan().term_device_hours);values.update(P=80,H_P=25,H_D=30)
        r=self.plan(term_device_hours=values,project_device_hour_ceiling=10000).review()
        self.assertEqual(r['full_study_components']['P_actual'],'80');self.assertEqual(len(r['component_limit_violations']),2)
        self.assertEqual(self.plan(project_device_hour_ceiling=None).review()['numerical_cost_status'],'no_project_device_ceiling')

    def test_optional_selection_requires_its_prices_and_one_extension_per_protocol(self):
        values=dict(self.plan().term_device_hours);values['C4']=None
        self.assertIn('C4',self.plan(term_device_hours=values,r4_selected=True).review()['missing_required_terms'])
        with self.assertRaises(ValueError):self.plan(selected_reuse=4)
        with self.assertRaises(ValueError):self.plan(selected_target='retrace')
        with self.assertRaises(ValueError):self.plan(corrective_device_hours=25)
        values['D']=13
        with self.assertRaises(ValueError):self.plan(term_device_hours=values,diagnostic_selected=True)
        with self.assertRaisesRegex(ValueError,'repeated'):
            self.plan(extensions=(ExtensionCeiling('b'*64,'delayed',1),ExtensionCeiling('b'*64,'branching',1)))
        r=self.plan(extensions=(ExtensionCeiling('b'*64,'delayed',None),)).review()
        self.assertIn('extension:'+'b'*64,r['missing_required_terms']);self.assertIsNone(r['full_study_device_hours'])

    def test_quantities_reject_bad_units_booleans_nonfinite_and_negative_numbers(self):
        for value in ('3600 seconds',True,-1,float('nan'),float('inf')):
            values=dict(self.plan().term_device_hours);values['C1']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.plan(term_device_hours=values)

    def test_references_and_plan_units_are_explicit_and_immutable(self):
        p=self.plan(evidence_sha256={});self.assertIn('H_N',p.review()['unreferenced_numeric_terms'])
        with self.assertRaises(TypeError):p.term_device_hours['H_N']=Decimal(0)
        with self.assertRaises(TypeError):p.evidence_sha256['H_N']='f'*64
        data=self.plan().to_dict();data['fixed']['confirmation_runs']=3
        with self.assertRaisesRegex(ValueError,'coefficients'):NFSPResourcePlan.from_dict(data)


class ResourceUsageTests(unittest.TestCase):
    def cpu(self,identity='one',scopes=('one',),seconds=100,category='control',state='actual'):
        return Usage(canonical(identity),category,'cpu_process_seconds',seconds,tuple(canonical(x) for x in scopes),canonical('evidence:'+identity),state)
    def gpu(self,identity,start,end,devices=('GPU-0',)):
        return Usage(canonical(identity),'accelerator','gpu_equipped_process_seconds',Decimal(end)-Decimal(start),
                     (canonical(identity),),canonical('evidence:'+identity),device_ids=devices,interval_start=start,interval_end=end)

    def test_actual_control_book_plus_live_reservation_reproduces_the_ceiling(self):
        r=measured_usage([self.cpu('recorded',('a','b'),'26991.557927068964025'),self.cpu('live',('c',),'15330',state='reserved')])
        control=r['cpu_categories']['control'];self.assertEqual(control['committed_process_seconds'],'42321.557927068964025')
        self.assertEqual(control['unallocated_seconds'],'878.442072931035975');self.assertTrue(control['within_ceiling'])
        self.assertFalse(r['project_consumption_complete']);self.assertEqual(r['physical_gpu_allocation_device_hours'],'0')

    def test_rollups_and_children_or_actual_and_reservation_cannot_overlap(self):
        for second in (self.cpu('child',('b',)),self.cpu('reservation',('a',),state='reserved'),self.cpu('other-category',('b',),category='verification')):
            with self.subTest(second=second.identity),self.assertRaisesRegex(ValueError,'double-count'):
                measured_usage([self.cpu('driver',('a','b')),second])
        item=self.cpu()
        with self.assertRaisesRegex(ValueError,'duplicate'):measured_usage([item,item])

    def test_over_budget_and_non_control_cpu_work_are_not_silently_discarded(self):
        r=measured_usage([self.cpu(seconds=44000),self.cpu('nn',('nn',),300,category='neural')])
        self.assertFalse(r['cpu_categories']['control']['within_ceiling']);self.assertEqual(r['cpu_categories']['control']['unallocated_seconds'],'-800')
        self.assertEqual(r['cpu_categories']['neural']['actual_process_seconds'],'300');self.assertIsNone(r['cpu_categories']['neural']['ceiling_seconds'])

    def test_shared_gpu_process_time_is_not_summed_as_physical_device_time(self):
        r=measured_usage([self.gpu('a','0','10'),self.gpu('b','5','15')])
        self.assertEqual(r['gpu_equipped_actual_process_seconds'],'20');self.assertEqual(r['known_physical_gpu_allocation_seconds_by_device'],{'GPU-0':'15'})
        self.assertEqual(Decimal(r['physical_gpu_allocation_device_hours']),Decimal(15)/3600)
        r=measured_usage([self.gpu(str(i),'0','3600') for i in range(8)])
        self.assertEqual(r['gpu_equipped_actual_process_seconds'],'28800');self.assertEqual(r['physical_gpu_allocation_device_hours'],'1')

    def test_multiple_devices_are_accounted_separately(self):
        r=measured_usage([self.gpu('a','0','10',('GPU-0','GPU-1')),self.gpu('b','5','15')])
        self.assertEqual(r['known_physical_gpu_allocation_seconds_by_device'],{'GPU-0':'15','GPU-1':'10'})
        self.assertEqual(Decimal(r['physical_gpu_allocation_device_hours']),Decimal(25)/3600)

    def test_unknown_gpu_intervals_are_unknown_not_elapsed_divided_by_concurrency(self):
        item=Usage(canonical('a'),'accelerator','gpu_equipped_process_seconds',3600,(canonical('a'),),canonical('proof'))
        r=measured_usage([item]);self.assertIsNone(r['physical_gpu_allocation_device_hours']);self.assertEqual(r['gpu_measurements_without_physical_intervals'],[item.identity])

    def test_invalid_units_intervals_or_scope_aliases_cannot_change_resource_dimensions(self):
        for change in ({'seconds':True},{'seconds':-1},{'unit':'device_hours'},{'category':'accelerator'},
                       {'covered_scopes':[]},{'covered_scopes':[canonical('a')]*2},{'interval_start':0}):
            data=self.cpu().to_dict();data.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):Usage(**data)
        data=self.gpu('a','0','10').to_dict();data['seconds']='1'
        with self.assertRaisesRegex(ValueError,'duration'):Usage(**data)

    def test_plan_and_usage_cli_are_read_only_and_protect_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=RouteLedgerTests().plan();plan=root/'plan.json';usage=root/'usage.json';output=root/'report.json'
            plan.write_text(json.dumps(p.to_dict()));usage.write_text(json.dumps({'version':VERSION,'measurements':[self.cpu().to_dict()]}))
            before=(plan.read_bytes(),usage.read_bytes())
            with redirect_stdout(io.StringIO()):self.assertEqual(main(['--plan',str(plan),'--usage',str(usage),'--output',str(output)]),0)
            self.assertEqual(before,(plan.read_bytes(),usage.read_bytes()));report=json.loads(output.read_text())
            self.assertEqual(report['source_sha256'],sha(Path(__file__).resolve().parents[1]/'tools/nfsp_resources.py'))
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['--plan',str(plan),'--output',str(plan)])
            self.assertEqual(before[0],plan.read_bytes())


if __name__=='__main__':unittest.main()
