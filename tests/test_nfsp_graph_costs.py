"""Synthetic arithmetic cannot qualify prices; real selected forwards check
the measurement boundary separately without invented large training clocks.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
import numpy as np
from test_nfsp_batch_costs import synthetic_report
from tools.nfsp_graph_costs import VERSION,GraphSetupObserver,actor_measurements,forecast,graph_io
from tools.audit_nfsp_graph_costs import recalculate,timing
from graygo.board_runtime import BoardRuntimeConfig,configure_runtime
from graygo.engine import State,resolve_reference
from graygo.frozen_forward import FrozenForwardPool
from graygo.models import QNet,PolicyNet
from graygo.nfsp import NFSPBackend


def fixture():
    old=synthetic_report();core=old['core_report'];core['plan']['runtime']['cuda_graphs']=True
    events=[{'kind':'weight_pack','seconds':.002,'pool':0,'model':'QNet','batch':32},
            {'kind':'capture','seconds':.03,'pool':0,'model':'QNet','batch':32}]
    actors={'cases':{},'requests_per_sample':32,'distinct_mixed_groups':32,'snapshots':16,'snapshot_copy_seconds':[.01]*16,
            'steady_actor_seconds_per_record':.01/32}
    for index,layout in enumerate(('shared_q','shared_average','distinct_mixed')):
        actors['cases'][layout]=[{'raw_seconds':.042 if index==0 and i==0 else .01,'steady_seconds':.01,
          'graph_setup_seconds':.032 if index==0 and i==0 else 0.,'graph_event_span':[0,2] if index==0 and i==0 else [2,2]} for i in range(4)]
    e=old['evaluation'];e['version']='graygo/batched-evaluation-cost-components/2'
    e['endpoint']['collection']=[{'status':'complete','physical_games':n,'committed_pairs':n//2} for n in (500,2000)]
    for timing in [case['timing'] for case in e['cases']]+[e['endpoint']['timing'],e['science']['timing']]:
        for group in timing['groups']:
            for wave in group['waves']:wave.update(raw_seconds=wave['seconds'],graph_setup_seconds=0.,graph_event_span=[2,2])
    setup={'events':events,'pools':{'0':{'captures':1,'weight_packs':1,'capture_seconds':.03,'weight_pack_seconds':.002,
        'resident_templates':1,'eager_fallbacks':0}},'capture_max_seconds':.03,'weight_pack_max_seconds':.002}
    io={'full_checkpoint_write_seconds':2.,'full_checkpoint_restore_seconds':3.,'public_payload_seconds':.01}
    r={'version':VERSION,'status':'complete','plan':dict(old['plan'],version=VERSION),'core_report':core,'evaluation':e,'actors':actors,
       'graph_io':io,'graph_setup':setup,'cold_start':old['cold_start'],'selected_model_load_seconds':old['selected_model_load_seconds']}
    r['forecast']=forecast(core,e,actors,io,setup,cold_seconds=old['cold_start']['seconds'],model_load_seconds=max(old['selected_model_load_seconds']))
    return r


class GraphCostArithmeticTests(unittest.TestCase):
    def test_cold_work_is_explicit_and_bounded_not_per_steady_action(self):
        r=fixture();result=recalculate(r);pilot=r['forecast']['per_seed']['pilot']
        self.assertEqual(pilot['snapshot_publication_upper'],15626)
        self.assertEqual(pilot['graph_capture_upper'],128+36)
        self.assertEqual(pilot['weight_pack_upper'],2*15626+32+36)
        self.assertEqual(pilot['components_seconds']['frozen_graph_captures'],164*.03)
        self.assertEqual(pilot['components_seconds']['frozen_parameter_packs'],31320*.002)
        self.assertEqual(pilot['components_seconds']['private_actors'],8000000*.01/32)
        self.assertEqual(result['capture_max_seconds'],.03)

    def test_zero_or_omitted_cold_cost_cannot_make_a_price_pass(self):
        for kind in ('frozen_graph_captures','frozen_parameter_packs'):
            r=fixture();r['forecast']['per_seed']['pilot']['components_seconds'].pop(kind)
            with self.subTest(kind=kind),self.assertRaises(ValueError):recalculate(r)
        r=fixture();r['actors']['cases']['shared_q'][0]['steady_seconds']=.042
        with self.assertRaises(ValueError):recalculate(r)

    def test_cold_event_cannot_be_subtracted_twice(self):
        r=fixture();sample=r['actors']['cases']['shared_q'][1]
        sample.update(raw_seconds=.042,graph_setup_seconds=.032,graph_event_span=[0,2])
        with self.assertRaisesRegex(ValueError,'twice'):recalculate(r)

    def test_changed_pool_counts_unmeasured_fallback_and_lost_events_rejected(self):
        for key,value in [('captures',0),('weight_packs',2),('eager_fallbacks',1),('resident_templates',65)]:
            r=fixture();r['graph_setup']['pools']['0'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):recalculate(r)
        r=fixture();r['graph_setup']['events'][0]['pool']=99
        with self.assertRaises(ValueError):recalculate(r)

    def test_cold_wave_partition_and_actual_work_are_checked(self):
        r=fixture();wave=r['evaluation']['cases'][0]['timing']['groups'][0]['waves'][0]
        wave['raw_seconds']+=.01
        with self.assertRaises(ValueError):recalculate(r)
        r=fixture();r['evaluation']['physical_rounds']-=1
        with self.assertRaises(ValueError):recalculate(r)

    def test_graph_startup_and_restore_counts_cannot_be_trimmed(self):
        r=fixture();r['forecast']['per_seed']['pilot']['graph_capture_upper']=64
        with self.assertRaises(ValueError):recalculate(r)
        r=fixture();r['forecast']['scientific_and_probe_components_seconds']['frozen_parameter_packs']=0.
        with self.assertRaises(ValueError):recalculate(r)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RealGraphCostTests(unittest.TestCase):
    def setUp(self):configure_runtime('cuda',True);torch.set_num_threads(1)

    def test_actual_private_graph_measurement_keeps_raw_setup_and_steady_intervals(self):
        config=BoardRuntimeConfig(device='cuda',envs=2,cuda_graphs=True)
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):backend=NFSPBackend(QNet(),PolicyNet(),config.backend_config(),device='cuda')
        states=[State(),resolve_reference(State(),0,1).state]
        with GraphSetupObserver().installed() as tracker:report=actor_measurements(config,backend,states,tracker)
        setup=tracker.report();self.assertTrue(setup['events']);self.assertGreater(setup['capture_max_seconds'],0)
        self.assertEqual(report['pool_stats']['captures'],4);self.assertEqual(report['pool_stats']['weight_packs'],4)
        for values in report['cases'].values():
            for sample in values:
                self.assertAlmostEqual(sample['raw_seconds'],sample['steady_seconds']+sample['graph_setup_seconds'])
                self.assertAlmostEqual(sample['graph_setup_seconds'],tracker.seconds_between(*sample['graph_event_span']))

    def test_failed_measurement_restores_original_forward_entry(self):
        original=FrozenForwardPool.forward
        with self.assertRaises(RuntimeError),GraphSetupObserver().installed():raise RuntimeError('injected')
        self.assertIs(FrozenForwardPool.forward,original)

    def test_real_graph_evidence_waves_partition_cold_work_without_fabricating_large_looks(self):
        from tools import nfsp_batch_costs as batch
        from graygo.curriculum import create_catalog
        import json
        base=Path(__file__).resolve().parents[1]
        data=json.loads((base/'docs/verification/probe-multicell-v2-construction/benchmark.json').read_text())
        states=[State.from_dict(data['cases'][0][which]['root']) for which in ('before','after')]
        config=BoardRuntimeConfig(device='cuda',envs=2,cuda_graphs=True)
        catalog=create_catalog(443,train_families=6,heldout_families=6)
        with tempfile.TemporaryDirectory() as directory,GraphSetupObserver().installed() as tracker:
            e=batch.evaluation_components(config,QNet().to('cuda'),PolicyNet().to('cuda'),states,catalog,Path(directory)/'e',
                inference_batch_size=4,pair_batch_size=2,full_look_games=16,reserve_bytes=0,cuda_graphs=True,setup_tracker=tracker)
            seen=set()
            for case in e['cases']:timing(case['timing'],tracker.events,max_pairs=2,limit=200,seen=seen)
            timing(e['endpoint']['timing'],tracker.events,max_pairs=2,limit=200,seen=seen)
            timing(e['science']['timing'],tracker.events,max_pairs=24,limit=200,seen=seen)
            self.assertEqual(e['physical_games'],160);self.assertEqual(e['endpoint']['looks'],[4,16])
            self.assertFalse(e['strength_claim']);self.assertTrue(seen)

    def test_selected_full_io_uses_graph_metadata_and_restores_original_factories(self):
        from test_nfsp_cost_profile import CostProfileTests
        from tools import nfsp_cost_profile as core
        from graygo.engine import Rules
        r,s,_=core.materialize(CostProfileTests().source())
        config=BoardRuntimeConfig(device='cuda',rules=Rules(3,6),envs=2,cuda_graphs=True)
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):backend=NFSPBackend(QNet(),PolicyNet(),config.backend_config(),device='cuda')
        old=(core.FrozenPolicies,core.FrozenNeuralPolicy)
        with tempfile.TemporaryDirectory() as directory:
            io=graph_io(config,backend,r,s,directory,reserve_bytes=0)
            payload=torch.load(Path(io['checkpoint'])/'cost-payload.pt',map_location='cpu',weights_only=True)
            self.assertTrue(all(v['frozen_execution']['backend']=='shared-static-weight-cuda-graph' for v in payload['frozen_snapshot_stress'].values()))
            self.assertFalse(io['training_resume_or_public_agent'])
        self.assertEqual((core.FrozenPolicies,core.FrozenNeuralPolicy),old)


if __name__=='__main__':unittest.main()
