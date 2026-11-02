"""Synthetic price arithmetic cannot qualify a run; real small source work
checks instrumentation separately, without fabricating a large evidence DB.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from graygo.board_runtime import BoardRuntimeConfig,configure_runtime
from graygo.curriculum import create_catalog
from graygo.models import ModelConfig,QNet,PolicyNet
from graygo.probe_benchmark import ProbeBenchmark
from tools.nfsp_batch_costs import Observer,evaluation_components,forecast,VERSION
from tools.audit_nfsp_batch_costs import timing,recalculate


def synthetic_group(pairs,rounds=2,tail=False):
    waves=[{'active_games':2*pairs,'seconds':.02}]
    if rounds>1:waves.append({'active_games':2 if tail else 2*pairs,'seconds':.01})
    n=sum(x['active_games'] for x in waves)
    return {'pairs':pairs,'physical_games':2*pairs,'waves':waves,'setup_seconds':.001,'seconds':.001+sum(x['seconds'] for x in waves),
            'physical_rounds':n,'completed_rounds':n}


def synthetic_timing(groups):
    checks=[.001,.002]
    return {'groups':groups,'integrity_seconds':checks,'metadata_seconds':.05,
            'seconds':sum(g['seconds'] for g in groups)+sum(checks)+.05}


def synthetic_report():
    base=Path(__file__).resolve().parents[1]
    core=json.loads((base/'docs/verification/nfsp-r4-full-cost/report.json').read_text())
    pair_batch=16;cases=[]
    for representation in ('q/random','q/tactical','average/random','average/tactical'):
        for mode in ('empty','dense','mixed-tail'):
            pairs=[]
            for i in range(pair_batch):
                rounds=1 if mode=='dense' or (mode=='mixed-tail' and i>0) else 2
                game={'rounds':rounds,'termination':'round_limit' if rounds==1 else 'mutual_pass'}
                pairs.append({'subject':[dict(game),dict(game)],'incumbent':[]})
            roots=[199]*pair_batch if mode!='empty' else [0]*pair_batch
            if mode=='mixed-tail':roots[0]=0
            g=synthetic_group(pair_batch,1 if mode=='dense' else 2,tail=mode=='mixed-tail')
            cases.append({'representation':representation,'mode':mode,'root_rounds_played':roots,'pairs':pairs,'timing':synthetic_timing([g])})
    endpoint=[]
    for games in (500,1500):
        remaining=games//2
        while remaining:
            count=min(pair_batch,remaining);endpoint.append(synthetic_group(count));remaining-=count
    science=[synthetic_group(24),synthetic_group(24)]
    all_base=[c['timing']['groups'][0] for c in cases]+endpoint
    e={'inference_batch_size':32,'pair_batch_size':pair_batch,'cases':cases,
        'endpoint':{'games':2000,'looks':[500,2000],'timing':synthetic_timing(endpoint)},
        'science':{'distinct_policy_objects':12,'collection':{'committed_blocks':2},'timing':synthetic_timing(science)},
        'probe_forward_seconds':[.004]*128,'integrity_seconds':[.001,.002]*14,
        'formal_learning_run':False,'strength_claim':False,
        'physical_games':sum(g['physical_games'] for g in all_base+science),'physical_rounds':sum(g['physical_rounds'] for g in all_base+science)}
    e['rates']={'baseline_wave':.02,'baseline_group_setup':.001,'endpoint_metadata':.05,'science_wave':.02,'science_group_setup':.001,
                'science_block_metadata_bound':.05,'policy_integrity':.002,'probe_forward':.004}
    plan={'version':VERSION,'runtime':core['plan']['runtime'],'source_sha256':core['plan']['source_sha256'],
          'inference_batch_size':32,'pair_batch_size':16,'margin':2.}
    return {'version':VERSION,'status':'complete','plan':plan,'core_report':core,'evaluation':e,
        'cold_start':{'seconds':2.,'result':{'source_sha256':plan['source_sha256'],'device':'cuda','cuda_initialized':True}},
        'selected_model_load_seconds':[.03,.04,.05],
        'forecast':forecast(core,e,margin=2,cold_seconds=2.,model_load_seconds=.05)}


class BatchedCostArithmeticTests(unittest.TestCase):
    def test_source_groups_are_counted_at_both_registered_look_boundaries(self):
        r=synthetic_report();x=recalculate(r);pilot=r['forecast']['per_seed']['pilot']
        self.assertEqual(x['groups_per_endpoint'],63);self.assertEqual(pilot['evaluation_groups'],1134)
        self.assertEqual(pilot['evaluation_endpoints'],18);self.assertEqual(pilot['model_loads_upper'],36)
        self.assertEqual(pilot['cold_process_starts_upper'],37)
        self.assertEqual(r['forecast']['scientific_blocks'],2000);self.assertEqual(r['forecast']['scientific_and_probe_model_loads_upper'],60)
        self.assertGreater(r['forecast']['H_D'],r['forecast']['H_P'])

    def test_tail_wave_rate_is_not_multiplied_by_every_games_maximum_rounds(self):
        r=synthetic_report();p=r['forecast']['per_seed']['pilot'];rate=r['evaluation']['rates']
        self.assertEqual(p['components_seconds']['evaluation_groups'],1134*(.001+200*.02))
        wrong=36000*200*.02
        self.assertLess(p['components_seconds']['evaluation_groups'],wrong/20)
        r['forecast']['per_seed']['pilot']['components_seconds']['evaluation_groups']=wrong
        with self.assertRaises(ValueError):recalculate(r)

    def test_missing_loads_cold_starts_and_checkpoint_terms_are_rejected(self):
        for component in ('evaluation_model_loads','cold_process_startups','checkpoints'):
            r=synthetic_report();r['forecast']['per_seed']['pilot']['components_seconds'].pop(component)
            with self.subTest(component=component),self.assertRaises(ValueError):recalculate(r)

    def test_incomplete_scientific_registry_and_round_growth_are_rejected(self):
        r=synthetic_report();r['evaluation']['science']['distinct_policy_objects']=1
        with self.assertRaises(ValueError):recalculate(r)
        r=synthetic_report();g=r['evaluation']['cases'][0]['timing']['groups'][0];g['waves'][1]['active_games']=g['physical_games']+2
        with self.assertRaises(ValueError):recalculate(r)

    def test_group_clock_and_mixed_root_accounting_cannot_hide_work(self):
        for change in ('clock','rounds','roots','probes','integrity'):
            r=synthetic_report()
            if change=='clock':r['evaluation']['cases'][0]['timing']['metadata_seconds']=0.
            if change=='rounds':r['evaluation']['cases'][0]['timing']['groups'][0]['physical_rounds']-=1
            if change=='roots':r['evaluation']['cases'][2]['root_rounds_played'][0]=199
            if change=='probes':r['evaluation']['probe_forward_seconds'].pop()
            if change=='integrity':r['evaluation']['integrity_seconds'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):recalculate(r)

    def test_changed_reuse_core_or_lower_study_total_cannot_be_substituted(self):
        r=synthetic_report();r['plan']['runtime']=dict(r['plan']['runtime'],q_reuse=1)
        with self.assertRaises(ValueError):recalculate(r)
        r=synthetic_report();r['forecast']['scientific_and_probe_device_hours']*=.5
        with self.assertRaises(ValueError):recalculate(r)

    def test_cpu_component_forecasts_never_become_device_hour_prices(self):
        r=synthetic_report();c=deepcopy(r['core_report']);c['plan']['runtime']['device']='cpu'
        value=forecast(c,r['evaluation'],margin=2,cold_seconds=2,model_load_seconds=.05)
        self.assertIsNone(value['H_P']);self.assertIsNone(value['H_N']);self.assertIsNone(value['scientific_and_probe_device_hours'])


class RealBatchedCostTests(unittest.TestCase):
    def test_actual_small_source_groups_sqlite_science_and_probes_leave_hooks_unchanged(self):
        import graygo.evaluation_batch as module
        configure_runtime('cpu',True);torch.set_num_threads(1);torch.manual_seed(53)
        base=Path(__file__).resolve().parents[1]
        benchmark=ProbeBenchmark.from_dict(json.loads((base/'docs/verification/probe-multicell-v2-construction/benchmark.json').read_text()))
        states=[benchmark.families[0].before.state,benchmark.families[0].after.state]
        config=BoardRuntimeConfig(width=8,blocks=1,norm_groups=2,device='cpu',envs=2)
        catalog=create_catalog(443,train_families=6,heldout_families=6)
        group=module.evaluate_pairs_batched;commit=module.commit_evaluation_actions
        with tempfile.TemporaryDirectory() as directory:
            e=evaluation_components(config,QNet(ModelConfig(8,1,2)),PolicyNet(ModelConfig(8,1,2)),states,catalog,Path(directory)/'profile',
                inference_batch_size=4,pair_batch_size=2,full_look_games=16,reserve_bytes=0)
            self.assertEqual(e['physical_games'],48+16+96);self.assertEqual(e['science']['distinct_policy_objects'],12)
            self.assertEqual(e['endpoint']['looks'],[4,16]);self.assertFalse(e['strength_claim'])
            for case in e['cases']:timing(case['timing'],max_pairs=2,limit=200)
            timing(e['science']['timing'],max_pairs=24,limit=200)
        self.assertIs(module.evaluate_pairs_batched,group);self.assertIs(module.commit_evaluation_actions,commit)

    def test_failure_restores_source_functions_and_does_not_suppress_the_error(self):
        import graygo.evaluation_batch as module
        from graygo.evaluation import FrozenNeuralPolicy
        group=module.evaluate_pairs_batched;commit=module.commit_evaluation_actions;frozen=FrozenNeuralPolicy.assert_frozen
        with self.assertRaisesRegex(RuntimeError,'injected'),Observer(torch.device('cpu')).installed():raise RuntimeError('injected')
        self.assertIs(module.evaluate_pairs_batched,group);self.assertIs(module.commit_evaluation_actions,commit)
        self.assertIs(FrozenNeuralPolicy.assert_frozen,frozen)


if __name__=='__main__':unittest.main()
