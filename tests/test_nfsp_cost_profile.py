from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.board_runtime import BoardRuntimeConfig
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.engine import Rules,State,resolve
from graygo.rule_state import RuleStateSpec,RuleObservation
from tools.nfsp_cost_profile import materialize,kernel_measurements,actor_measurements,dense_collection,io_measurements,forecast,pipeline_measurements,profile


class CostProfileTests(unittest.TestCase):
    def source(self,capacity=16,records=24):
        rules=Rules(3,6);spec=RuleStateSpec(rules);buffer=ReplayBuffer(capacity,spec);rng=np.random.default_rng(19)
        state=State(rules);episode=0
        for record in range(0,records,2):
            b=int(rng.choice(state.legal_actions(1)));w=int(rng.choice(state.legal_actions(2)));next_=resolve(state,b,w).state
            reward=0.
            if next_.done:
                from graygo.engine import score
                reward=float(score(next_.board,3).utility_black)
            for role,action in ((1,b),(2,w)):
                buffer.add(RuleObservation(state,role),action,reward*(1 if role==1 else -1),RuleObservation(next_,role),next_.done,
                    behavior_prob=1/len(state.legal_actions(role)),stream=2*episode+role-1,step=state.rounds_played)
            state=next_
            if state.done:episode+=1;state=State(rules)
        return buffer

    def config(self):
        return BoardRuntimeConfig(rules=Rules(3,6),envs=2,seed=11,batch=8,learning_start=8,capacity=16,
            q_reuse=4,target_every=16,width=8,blocks=1,norm_groups=2,device='cpu')

    def test_materialization_retains_real_transitions_without_inventing_old_clocks(self):
        source=self.source();source.q_insertions=None;before={k:v.copy() for k,v in source.arrays.items()}
        replay,reservoir,report=materialize(source)
        self.assertEqual(replay.insertions,replay.q_insertions);self.assertEqual(replay.insertions,16)
        np.testing.assert_array_equal(replay.arrays['ordinal'],np.arange(16))
        self.assertTrue(report['sequence_links_retained']);self.assertGreater(report['retained_links_remapped'],0)
        for i in np.flatnonzero(replay.arrays['next_index']>=0):
            following=int(replay.arrays['next_index'][i]);self.assertEqual(replay.arrays['next_ordinal'][i],following)
            self.assertEqual(replay.arrays['stream'][i],replay.arrays['stream'][following])
            self.assertEqual(replay.arrays['step'][i]+1,replay.arrays['step'][following])
        self.assertFalse(report['source_historical_q_count_known']);self.assertEqual(report['source_original_insertions'],24)
        self.assertEqual(report['new_environment_actions'],0);self.assertEqual(reservoir.size,16)
        targets=reservoir.sample(16,np.random.default_rng(1))
        self.assertTrue((targets['targets'].sum(-1)==1).all());self.assertFalse((targets['targets']*~targets['legal']).any())
        for name,array in before.items():np.testing.assert_array_equal(array,source.arrays[name])
        source.arrays['value_mask'][0]=False
        with self.assertRaisesRegex(ValueError,'valid Q starts'):materialize(source)

    def test_real_kernels_use_all_nonterminal_full_buffer_targets_and_count_updates(self):
        torch.set_num_threads(1);r,s,_=materialize(self.source());backend,report=kernel_measurements(self.config(),r,s,iterations=2,warmup=1)
        self.assertEqual(report['actual_optimizer_updates_each'],3);self.assertEqual(report['measured_q_loss_targets'],16)
        self.assertEqual(report['new_fresh_learner_records'],0)
        self.assertEqual(backend.counters.fresh_records,0);self.assertEqual(backend.counters.q_updates,0)
        self.assertEqual(len(report['q_seconds']),2);self.assertTrue(all(t>0 for t in report['supervised_seconds']))

    def test_paired_actor_stress_has_distinct_immutable_versions_and_real_private_actions(self):
        torch.set_num_threads(1);r,s,_=materialize(self.source());backend,_=kernel_measurements(self.config(),r,s,iterations=1,warmup=0)
        states=[r.spec.codec.unpack(r.arrays['spatial'][i])[0] for i in (0,2)]
        report=actor_measurements(self.config(),backend,states,iterations=2)
        self.assertEqual(report['requests_per_sample'],4);self.assertEqual(report['distinct_mixed_groups'],4)
        self.assertEqual(set(report['samples_seconds']),{'shared_q','shared_average','distinct_mixed'})
        self.assertEqual(len(report['snapshot_copy_seconds']),2);self.assertTrue(report['paired_public_states'])

    def test_real_full_buffer_append_uses_explicit_offsets_and_local_collector_counts(self):
        torch.set_num_threads(1);r,s,_=materialize(self.source());backend,_=kernel_measurements(self.config(),r,s,iterations=1,warmup=0)
        states=[r.spec.codec.unpack(r.arrays['spatial'][i])[0] for i in (0,2)]
        old_max=int(r.arrays['stream'].max());report=dense_collection(self.config(),backend,r,s,states,records=8)
        self.assertEqual(report['new_profile_records'],8);self.assertEqual(report['collector_local_records'],8)
        self.assertEqual(report['buffer_record_offset'],16);self.assertEqual(r.insertions,24)
        self.assertGreater(report['new_stream_offset'],old_max);self.assertEqual(r.size,s.size)
        self.assertEqual(backend.counters.fresh_records,0)

    def test_full_io_roundtrip_is_not_a_public_or_runtime_training_checkpoint(self):
        torch.set_num_threads(1);r,s,_=materialize(self.source());backend,_=kernel_measurements(self.config(),r,s,iterations=1,warmup=0)
        with tempfile.TemporaryDirectory() as directory:
            report=io_measurements(self.config(),backend,r,s,directory,reserve_bytes=0)
            self.assertTrue(report['buffer_roundtrip_verified']);self.assertTrue(report['model_roundtrip_verified'])
            self.assertFalse(report['training_resume_or_public_agent'])
            path=Path(report['checkpoint']);self.assertTrue((path/'cost-payload.pt').is_file())
            self.assertFalse((path/'runtime.pt').exists());self.assertFalse((path/'models.pt').exists())

    def rates(self):
        return dict(collection_per_record=.001,actor_per_record=.002,snapshot=.01,q_update=.1,supervised_update=.05,
            target_copy=.002,checkpoint_write=1.,checkpoint_restore=1.,baseline_eval_round=.001,neural_eval_round=.001,
            public_archive=.1,public_model_load=.2,policy_integrity=.001,setup=2.)

    def test_forecast_uses_exact_reuse_refresh_and_pilot_prefix_counts(self):
        config=BoardRuntimeConfig(q_reuse=4,device='cuda');games={'pilot':36000,'screening_cumulative':40000,'confirmation':44000}
        report=forecast(config,self.rates(),margin=2,evaluation_games=games);pilot=report['per_seed']['pilot']
        self.assertEqual(pilot['q_updates'],62472);self.assertEqual(pilot['supervised_update_upper'],5860)
        self.assertEqual(pilot['target_refreshes'],625);self.assertEqual(pilot['full_checkpoint_count_upper'],86)
        self.assertGreater(report['H_D'],report['H_P']);self.assertGreater(report['H_N'],report['H_D'])
        self.assertEqual(report['concurrency'],1)
        r1=forecast(replace(config,q_reuse=1),self.rates(),margin=2,evaluation_games=games)
        self.assertEqual(r1['per_seed']['pilot']['target_refreshes'],625)
        self.assertEqual(r1['per_seed']['pilot']['q_updates']*4,pilot['q_updates'])

    def test_cpu_times_do_not_become_cuda_device_hour_prices(self):
        r=forecast(BoardRuntimeConfig(device='cpu'),self.rates(),margin=2,
            evaluation_games={'pilot':36000,'screening_cumulative':40000,'confirmation':44000})
        self.assertIsNone(r['H_P']);self.assertIsNone(r['H_N']);self.assertIsNone(r['scientific_and_probe_device_hours'])
        self.assertGreater(r['per_seed']['confirmation']['allocated_process_hours'],0)

    def test_missing_components_invalid_margins_and_screening_reset_are_rejected(self):
        games={'pilot':10,'screening_cumulative':9,'confirmation':10}
        with self.assertRaisesRegex(ValueError,'include its pilot'):forecast(BoardRuntimeConfig(),self.rates(),margin=2,evaluation_games=games)
        for variant in ('missing','negative','nan'):
            rates=self.rates()
            if variant=='missing':del rates['snapshot']
            elif variant=='negative':rates['snapshot']=-1
            else:rates['snapshot']=float('nan')
            with self.subTest(variant=variant),self.assertRaises(ValueError):forecast(BoardRuntimeConfig(),rates,margin=2,evaluation_games={'pilot':1,'screening_cumulative':1,'confirmation':1})

    def tiny_pipeline(self,config,root,*,records,time_cap_seconds):
        import graygo.board_profile as module
        phases={};root=Path(root);root.mkdir(parents=True,exist_ok=True)
        for name,cfg in (('closed',config),('early_open_diagnostic',replace(config,reservoir_open=0,handoff_end=2))):
            runtime=module.BoardNFSPRuntime(cfg);result=runtime.run_profile(records,time_cap_seconds=60)
            self.assertEqual(result['status'],'complete');result['setup_seconds']=.001;phases[name]=result
        report={'status':'complete','phases':phases}
        import json
        (root/'report.json').write_text(json.dumps(report));return report

    def test_source_pipeline_instrumentation_preserves_real_clocks_and_restores_functions(self):
        import graygo.board_collection as collector_module
        import graygo.board_profile as profile_module
        original=collector_module.commit_many;runtime_class=profile_module.BoardNFSPRuntime
        with tempfile.TemporaryDirectory() as directory,patch.object(profile_module,'profile_board',side_effect=self.tiny_pipeline):
            report,observers,rate=pipeline_measurements(self.config(),directory,records=64,time_cap_seconds=60)
        self.assertEqual(len(observers),2);self.assertGreater(rate,0)
        self.assertGreater(observers[0]['actor_seconds'],0);self.assertGreater(observers[1]['snapshot_copies'],0)
        self.assertEqual(report['phases']['closed']['counters']['fresh_records'],64)
        self.assertIs(collector_module.commit_many,original);self.assertIs(profile_module.BoardNFSPRuntime,runtime_class)

    def test_complete_diagnostic_orchestration_cannot_become_a_cuda_campaign_price(self):
        import json
        from graygo.storage import CheckpointStore
        source=self.source();states=[source.spec.codec.unpack(source.arrays['spatial'][i])[0] for i in (0,2)]
        benchmark=SimpleNamespace(family_coverage_complete=True,sha256='a'*64,
            families=[SimpleNamespace(before=SimpleNamespace(state=states[0]),after=SimpleNamespace(state=states[1]))])
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);store=CheckpointStore(base/'input',reserve_bytes=0)
            bundle=store.save(24,lambda path:source.save(path/'replay.npz'),metadata={'fixture':True},estimated_peak_bytes=65536)
            with patch('graygo.board_profile.profile_board',side_effect=self.tiny_pipeline):
                result=profile(self.config(),bundle,benchmark,base/'profile',records=64,iterations=1,warmup=0,
                    actor_iterations=1,dense_records=8,margin=2,time_cap_seconds=120,diagnostic=True,reserve_bytes=0,
                    evaluation_games={'pilot':2,'screening_cumulative':4,'confirmation':6})
            self.assertEqual(result['status'],'complete');self.assertTrue(result['full_component_coverage'])
            self.assertFalse(result['candidate_price_forecast_available']);self.assertFalse(result['execution_authorized'])
            self.assertIsNone(result['forecast']['H_N']);self.assertEqual(result['fresh_profile_records'],136)
            self.assertTrue((base/'profile/report.json').is_file())
            with self.assertRaisesRegex(ValueError,'exists'):profile(self.config(),bundle,benchmark,base/'profile',records=64,
                iterations=1,warmup=0,actor_iterations=1,dense_records=8,margin=2,time_cap_seconds=120,diagnostic=True,reserve_bytes=0)


if __name__=='__main__':unittest.main()
