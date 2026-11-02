"""Complete transition/rank/RNG checks against retained literal v1 oracles."""
import ast
from dataclasses import FrozenInstanceError
import hashlib
import inspect
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from graygo import agents
from graygo.agents import tactical,_tactical_reference,_tactical_ranks
from graygo.engine import BLACK,WHITE,GRAY,Rules,State,RoundRecord,Group,PlacementResolver,groups,resolve,resolve_reference,transform
from graygo.evaluation import BaselinePolicy,EpisodeMixture,evaluate_pair


def reference_ranks(state,player):
    atari=frozenset(point for group in groups(state.board,state.rules.size)
                    if group.color==player and len(group.liberties)==1 for point in group.stones)
    result=[]
    from graygo.engine import neighbors
    for action in state.legal_actions(player):
        if action==state.rules.pass_action:continue
        transition=resolve_reference(state,action,state.rules.pass_action) if player==BLACK else resolve_reference(state,state.rules.pass_action,action)
        captures=sum(len(group.stones) for group in transition.captured if group.color!=player)
        own=next((g for g in groups(transition.state.board,state.rules.size) if action in g.stones and g.color==player),None)
        safe=own is not None and len(own.liberties)>=2;saved=len(own.stones & atari) if safe else 0
        eye=all(state.board[p]==player for p in neighbors(state.rules.size)[action])
        if captures:result.append(((3,captures),action))
        elif saved:result.append(((2,saved),action))
        elif safe and not eye:result.append(((1,0),action))
    return result


def body_sha(function,skip_doc=False):
    def normalized(value):
        if isinstance(value,ast.AST):
            return [type(value).__name__,{key:normalized(v) for key,v in ast.iter_fields(value)
                if (v is not None and v!=[]) or (isinstance(value,ast.Constant) and key=='value')}]
        if isinstance(value,list):return [normalized(x) for x in value]
        return value
    body=ast.parse(inspect.getsource(function)).body[0].body
    if skip_doc:body=body[1:]
    return hashlib.sha256(json.dumps(normalized(body),sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PreparedPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'docs/verification/probe-multicell-v2-construction/benchmark.json'
        data=json.loads(path.read_text())
        cls.probe_states=tuple(State.from_dict(case[which]['root']) for case in data['cases'] for which in ('before','after'))

    def assert_state(self,state,*,choices=True):
        before=state.canonical_json()
        for player in (BLACK,WHITE):
            prepared=PlacementResolver(state,player)
            for action in state.legal_actions(player):
                expected=resolve_reference(state,action,state.rules.pass_action) if player==BLACK else resolve_reference(state,state.rules.pass_action,action)
                actual=prepared.resolve(action)
                self.assertEqual(actual.transition,expected)
                self.assertEqual(resolve(state,expected.black,expected.white),expected)
                own=next((g for g in groups(expected.state.board,state.rules.size) if action in g.stones and g.color==player),None)
                self.assertEqual(actual.placed_group,own if action!=state.rules.pass_action else None)
                self.assertEqual(State.from_dict(actual.transition.state.to_dict()),actual.transition.state)
            self.assertEqual(_tactical_ranks(state,player),reference_ranks(state,player))
            if choices:
                for seed in (0,17,4294967297):
                    a,b=random.Random(seed),random.Random(seed)
                    self.assertEqual(tactical(state,player,a),_tactical_reference(state,player,b))
                    self.assertEqual(a.getstate(),b.getstate())
        self.assertEqual(state.canonical_json(),before)

    def test_literal_reference_bodies_are_unchanged_from_frozen_v1(self):
        self.assertEqual(body_sha(resolve_reference),'9163ad71918e6fb10f6ce9dc23777ad158aab6a57863f0b003daf4a73442f204')
        self.assertEqual(body_sha(_tactical_reference,True),'a4f9a96184c12423c8cf7e8edc739d4817948c345a044602e2d6444a7f5692a1')
        self.assertIs(agents.resolve,resolve_reference)
        self.assertEqual(agents.BASELINE_VERSION,'graygo/baselines/1')

    def test_all_frozen_dense_probe_roots_match_complete_transition_group_and_ranks(self):
        self.assertEqual(len(self.probe_states),64)
        for state in self.probe_states:self.assert_state(state)

    def test_actual_games_restrictions_history_terminal_and_both_collision_rules(self):
        rng=random.Random(412)
        for rule in ('gray','remove'):
            state=State(Rules(3,12,rule))
            for index in range(100):
                if state.done:state=State(state.rules)
                self.assert_state(state)
                state=resolve_reference(state,rng.choice(state.legal_actions(BLACK)),rng.choice(state.legal_actions(WHITE))).state
                if index%7==0 and not state.done:
                    blocked=frozenset(p for p in range(state.rules.points) if rng.random()<.4)
                    variant=State(state.rules,state.board,blocked,state.forbidden_white,state.history,state.pass_streak,state.rounds_played)
                    self.assert_state(variant)
            one=resolve_reference(State(Rules(3,4,rule)),9,9).state
            self.assert_state(one)
            ended=resolve_reference(one,9,9).state
            for function in (resolve,resolve_reference):
                with self.assertRaises(ValueError):function(ended,9,9)
            with self.assertRaises(ValueError):PlacementResolver(ended,BLACK)

    def test_large_board_wrapping_and_color_symmetries_preserve_full_outcomes(self):
        rng=random.Random(38)
        for size in (9,13,19):
            state=State(Rules(size,40))
            for _ in range(30):state=resolve_reference(state,rng.choice(state.legal_actions(BLACK)),rng.choice(state.legal_actions(WHITE))).state
            self.assert_state(state,choices=False)
            if size==9:
                for rotation in range(4):self.assert_state(transform(state,rotation=rotation,shift=(2,5),reflect=True,swap=True),choices=False)

    def test_capture_dedup_gray_and_friendly_suicide_with_no_change_and_ko(self):
        # A toroid-filling friendly group loses its final liberty and suicides.
        for player in (BLACK,WHITE):
            board=[player]*9;board[0]=0
            self.assert_state(State(Rules(3,2),board))
        # Gray atari can touch the same candidate several ways; capture once.
        board=[GRAY]*9;board[0]=0
        self.assert_state(State(Rules(3,1),board))
        # Noncapturing singleton suicide, accumulated bans and ko next-move bans.
        board=[0]*9
        for p in (1,2,3,6):board[p]=WHITE
        base=State(Rules(3,6),board)
        first=resolve_reference(base,0,9).state;self.assert_state(first)
        for record in (RoundRecord(base.board,4,9),RoundRecord(base.board,9,5)):
            state=State(base.rules,base.board,frozenset((8,)),frozenset((7,)),(record,),0,1)
            self.assert_state(state)

    def test_prepared_analysis_is_immutable_and_invalid_actions_stay_rejected(self):
        state=State(Rules(3,4));prepared=PlacementResolver(state,BLACK)
        with self.assertRaises(FrozenInstanceError):prepared.player=WHITE
        for action in (-1,10,True,1.5):
            with self.assertRaises(ValueError):prepared.resolve(action)
            with self.assertRaises(ValueError):resolve(state,action,9)
        occupied=resolve_reference(state,0,1).state
        with self.assertRaises(ValueError):PlacementResolver(occupied,BLACK).resolve(0)
        blocked=State(state.rules,state.board,frozenset((0,)))
        with self.assertRaises(ValueError):PlacementResolver(blocked,BLACK).resolve(0)
        # Existing equal-to-player scalar spellings normalize to canonical ints.
        self.assertEqual(PlacementResolver(state,True).resolve(0),prepared.resolve(0))
        for player in (0,3):
            with self.assertRaises(ValueError):PlacementResolver(state,player)

    def test_paired_games_and_episode_mixtures_keep_actions_scores_and_private_ties(self):
        class ReferencePolicy(BaselinePolicy):
            def begin_episode(self,seed):
                actor=super().begin_episode(seed);actor.function=_tactical_reference;return actor
        fast=BaselinePolicy('tactical');reference=ReferencePolicy('tactical');opponent=BaselinePolicy('random')
        self.assertEqual(fast.sha256,reference.sha256)
        for seed in range(10):
            root=State(Rules(3,12))
            self.assertEqual(evaluate_pair(fast,opponent,root,unit_seed=seed),evaluate_pair(reference,opponent,root,unit_seed=seed))
            a=EpisodeMixture((fast,opponent),(.6,.4));b=EpisodeMixture((reference,opponent),(.6,.4))
            self.assertEqual(evaluate_pair(a,opponent,root,unit_seed=seed),evaluate_pair(b,opponent,root,unit_seed=seed))

    def test_baseline_only_batched_cli_runs_without_importing_optional_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);confidence=root/'confidence.json'
            confidence.write_text(json.dumps({'name':'core-batch','endpoints':['u'],'looks':[4],'alpha':.05}))
            arguments=['evaluate','--run-dir',str(root/'run'),'--confidence-plan',str(confidence),'--name','core-batch',
                '--subject-label','tactical','--endpoint','u','--subject','tactical','--opponent','random','--games','4',
                '--size','3','--limit','3','--time-cap-hours','.01','--pair-batch-size','2']
            code='''import importlib.abc,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise ModuleNotFoundError('optional Torch deliberately blocked')
sys.meta_path.insert(0,NoTorch())
from graygo.cli import main
result=main(sys.argv[1:]);assert 'torch' not in sys.modules;raise SystemExit(result)
'''
            result=subprocess.run([sys.executable,'-c',code,*arguments],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['result']['collection']['physical_games'],4)


if __name__=='__main__':unittest.main()
