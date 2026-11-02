from copy import deepcopy
import unittest
from graygo.pilot_gates import PilotProtocol,ScreeningProtocol,DEVELOPMENT_SEEDS,PRE,POST,endpoint,subject_label


class PilotTests(unittest.TestCase):
    def protocol(self,diagnostics=True):return PilotProtocol('pilot','a'*64,'b'*64,{s:f'{s:064x}' for s in DEVELOPMENT_SEEDS},diagnostics)
    def read(self,p,seed,record,opponent,*,representation='greedy_q',games=500,value=1.):
        family=p.family;key=endpoint(seed,record,representation,opponent)
        r=family.estimate(key,games,[value]*(games//2))
        r.update(comparison='utility',available_games_per_pairing=games,representation=representation,
            subject_label=subject_label(seed,record,representation),subject_sha256=f'{seed+record:064x}',opponent_sha256='a'*64 if opponent=='random' else 'b'*64,
            evidence_sha256='c'*64,plan_sha256='d'*64,subject_origin={'training_run_sha256':f'{seed:064x}','seed':seed,'records':record,'rules':{'size':9,'max_rounds':200,'collision':'gray'}},
            rules={'size':9,'max_rounds':200,'collision':'gray'},opening_kind='official-empty')
        return r
    def all_reads(self,p):
        data=[self.read(p,s,n,o) for s in DEVELOPMENT_SEEDS for n in (*PRE,*POST) for o in ('random','tactical')]
        if p.include_average_diagnostics:data += [self.read(p,s,n,o,representation='average') for s in DEVELOPMENT_SEEDS for n in POST for o in ('random','tactical')]
        return data

    def test_all_seeds_both_windows_and_average_diagnostics_required(self):
        p=self.protocol();reads=self.all_reads(p);result=p.cohort(reads)
        self.assertEqual(result['status'],'pass');self.assertFalse(result['a2_complete']);self.assertFalse(result['deployed_strength_established'])
        no_average=[r for r in reads if r['representation']=='greedy_q']
        self.assertEqual(p.cohort(no_average)['status'],'incomplete')
        missing=[r for r in reads if not r['endpoint'].startswith('seed102/')]
        self.assertEqual(p.cohort(missing)['status'],'incomplete')

    def test_two_of_three_uses_complete_same_checkpoint_opponent_passes(self):
        p=self.protocol(False);reads=[]
        for n in PRE:
            for o in ('random','tactical'):
                # Each snapshot fails at least one opponent, despite passes elsewhere.
                value=1. if (n==PRE[0] and o=='random') or (n==PRE[1] and o=='tactical') else -1.
                reads.append(self.read(p,100,n,o,games=2000,value=value))
        self.assertEqual(p.window(reads,100,window='pre')['status'],'no_go')
        reads=[self.read(p,100,n,o,games=2000,value=1. if n in PRE[:2] else -1.) for n in PRE for o in ('random','tactical')]
        self.assertEqual(p.window(reads,100,window='pre')['status'],'pass')
        opened=p.label_insertion(reads,100,5_000_000,scope='development')
        self.assertTrue(opened['allowed'])
        self.assertFalse(p.label_insertion(reads,100,6_000_000,scope='development')['allowed'])
        self.assertTrue(p.label_insertion(reads,100,6_000_000,scope='development',opening_event=opened['opening_event'])['allowed'])
        self.assertFalse(p.label_insertion(reads,100,4_999_999,scope='development')['allowed'])

    def test_latest_look_checkpoint_origin_and_support_are_enforced(self):
        p=self.protocol(False);reads=[self.read(p,100,n,o) for n in PRE for o in ('random','tactical')]
        bad=deepcopy(reads);bad[0]['available_games_per_pairing']=2000
        with self.assertRaises(ValueError):p.window(bad,100,window='pre')
        bad=deepcopy(reads);bad[0]['subject_origin']['records']=5_000_000
        with self.assertRaises(ValueError):p.window(bad,100,window='pre')
        bad=deepcopy(reads);bad[0]['opening_kind']='fixed-public-opening-catalog'
        with self.assertRaises(ValueError):p.window(bad,100,window='pre')
        bad=deepcopy(reads);bad[0]['lower']=1.
        with self.assertRaises(ValueError):p.window(bad,100,window='pre')

    def test_two_passes_do_not_require_a_third_cherry_picked_maximum_look(self):
        p=self.protocol(False)
        reads=[self.read(p,100,n,o,value=1. if n in PRE[:2] else 0.) for n in PRE for o in ('random','tactical')]
        self.assertEqual(p.window(reads,100,window='pre')['status'],'pass')
        missing=[r for r in reads if '/5000000/' not in r['endpoint']]
        self.assertEqual(p.window(missing,100,window='pre')['status'],'incomplete')
        with self.assertRaises(ValueError):PilotProtocol('bad','a'*64,'b'*64,{s:f'{s:064x}' for s in DEVELOPMENT_SEEDS},alpha=.1)

    def test_saved_protocols_roundtrip_and_reject_changed_thresholds_or_seeds(self):
        p=self.protocol();self.assertEqual(PilotProtocol.from_dict(p.to_dict()),p)
        bad=deepcopy(p.to_dict());bad['pre_records']=[4_000_000,4_500_000,5_000_000]
        with self.assertRaises(ValueError):PilotProtocol.from_dict(bad)
        screen=ScreeningProtocol('screen','a'*64,'b'*64,{s:f'{s:064x}' for s in DEVELOPMENT_SEEDS})
        self.assertEqual(ScreeningProtocol.from_dict(screen.to_dict()).to_dict(),screen.to_dict())
        bad=deepcopy(screen.to_dict());bad['thresholds']['tactical_lower_at_least']=0.
        with self.assertRaises(ValueError):ScreeningProtocol.from_dict(bad)

    def test_confirmation_is_fixed_and_does_not_performance_filter_seeds(self):
        p=self.protocol(False)
        for seed in (200,201,202):
            self.assertFalse(p.label_insertion([],seed,4_999_999,scope='confirmation')['allowed'])
            self.assertTrue(p.label_insertion([],seed,5_000_000,scope='confirmation')['allowed'])
        with self.assertRaises(ValueError):p.label_insertion([],100,5_000_000,scope='confirmation')

    def test_batched_execution_is_preregistered_and_cannot_be_changed_by_receipts(self):
        from graygo.evaluation import pair_execution_identity
        settings=dict(inference_batch_size=8,pair_batch_size=4,inference_device='cuda')
        p=PilotProtocol('batch','a'*64,'b'*64,{s:f'{s:064x}' for s in DEVELOPMENT_SEEDS},**settings)
        self.assertEqual(PilotProtocol.from_dict(p.to_dict()),p)
        reads=self.all_reads(p)
        for read in reads:read.update(subject_inference=p.execution_settings()['neural_inference'],evaluation_execution=pair_execution_identity(4))
        self.assertEqual(p.cohort(reads)['status'],'pass')
        bad=deepcopy(reads);bad[0]['subject_inference']['batch_size']=4
        with self.assertRaisesRegex(ValueError,'inference layout'):p.cohort(bad)
        bad=deepcopy(reads);bad[0]['evaluation_execution']=pair_execution_identity(1)
        with self.assertRaisesRegex(ValueError,'pair batching'):p.cohort(bad)
        screen=ScreeningProtocol('screen','a'*64,'b'*64,{s:f'{s:064x}' for s in DEVELOPMENT_SEEDS},**settings)
        self.assertEqual(ScreeningProtocol.from_dict(screen.to_dict()).to_dict(),screen.to_dict())
        rows=[self.read(screen,s,20000000,o,representation='average',games=2000) for s in DEVELOPMENT_SEEDS for o in ('random','tactical')]
        for row in rows:row.update(subject_inference=screen.pilot.execution_settings()['neural_inference'],evaluation_execution=pair_execution_identity(4))
        self.assertEqual(screen.decide(rows)['status'],'pass')
        rows[0].pop('subject_inference')
        with self.assertRaises(ValueError):screen.decide(rows)

    def test_screening_grades_average_with_explicit_tactical_bar(self):
        p=ScreeningProtocol('screen','a'*64,'b'*64,{s:f'{s:064x}' for s in DEVELOPMENT_SEEDS})
        reads=[self.read(p,s,20_000_000,o,representation='average',games=2000,value=1. if o=='random' else .2) for s in DEVELOPMENT_SEEDS for o in ('random','tactical')]
        self.assertEqual(p.decide(reads)['status'],'pass')
        bad=[self.read(p,s,20_000_000,o,representation='average',games=2000,value=1. if o=='random' else .1) for s in DEVELOPMENT_SEEDS for o in ('random','tactical')]
        self.assertEqual(p.decide(bad)['status'],'no_go')
        bad=deepcopy(reads);bad[0]['representation']='greedy_q'
        with self.assertRaises(ValueError):p.decide(bad)

if __name__=='__main__':unittest.main()
