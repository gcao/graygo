"""Concurrent frozen games with independent private commitments and RNGs.

Only public observations enter grouped forwards. Each game retains the scalar
evaluator's seeds, opening, color exchange, resolver, score and trajectory hash.
Neural policy identities declare their fixed padded numerical forward shape.
"""
from dataclasses import dataclass
import hashlib
import json

from .engine import BLACK,WHITE,State,resolve,score
from .evaluation import _NeuralActor,seed_for

BATCH_EVALUATION_VERSION='graygo/frozen-pair-batching/1'
MAX_PAIRS=32


@dataclass(frozen=True)
class PairRequest:
    subject: object
    opponent: object
    root: State
    unit_seed: int
    incumbent: object|None=None

    def __post_init__(self):
        if not isinstance(self.root,State) or self.root.done:raise ValueError('complete nonterminal evaluation root required')
        if type(self.unit_seed) is not int or self.unit_seed<0:raise ValueError('nonnegative private evaluation seed required')


def commit_evaluation_actions(requests):
    """Group immutable model work, then draw each actor's private RNG once.

    The caller supplies only (actor, public state, player). Computed commitments
    are never an input to another actor. Baselines and external scalar actors
    retain their own action API. Grouping uses object identity, not a model hash
    that could conflate different devices or mutable objects.
    """
    requests=tuple(requests);groups={};weights={}
    for index,(actor,state,player) in enumerate(requests):
        if state.done or player not in (BLACK,WHITE):raise ValueError('live public state/player required')
        if isinstance(actor,_NeuralActor):groups.setdefault(id(actor.policy),[]).append(index)
    for indices in groups.values():
        policy=requests[indices[0]][0].policy
        values=policy._action_weights_many([(requests[i][1],requests[i][2]) for i in indices])
        if len(values)!=len(indices):raise ValueError('grouped inference lost an actor')
        weights.update(zip(indices,values))
    actions=[]
    for index,(actor,state,player) in enumerate(requests):
        action=actor._draw(weights[index]) if index in weights else actor.action(state,player)
        if action not in state.legal_actions(player):raise ValueError('evaluation policy chose an illegal move')
        actions.append(action)
    return actions


class _Game:
    def __init__(self,subject,opponent,root,player,subject_seed,opponent_seed):
        self.subject=subject.begin_episode(subject_seed);self.opponent=opponent.begin_episode(opponent_seed)
        self.state=root;self.player=player;self.rounds=0
        self.hash=hashlib.sha256(root.canonical_json().encode());self.result=None

    def advance(self,mine,theirs):
        black,white=(mine,theirs) if self.player==BLACK else (theirs,mine)
        self.hash.update(json.dumps([black,white],separators=(',',':')).encode())
        self.state=resolve(self.state,black,white).state;self.rounds+=1
        if self.state.done:
            scored=score(self.state.board,self.state.rules.size);black=float(scored.utility_black)
            self.result={'subject_utility':black if self.player==BLACK else -black,'black_utility':black,'rounds':self.rounds,
                'termination':self.state.terminal_reason,'black_score_units':scored.black_units,'white_score_units':scored.white_units,
                'trajectory_sha256':self.hash.hexdigest(),'subject_component':getattr(self.subject,'component',None),
                'opponent_component':getattr(self.opponent,'component',None)}


def evaluate_pairs_batched(requests):
    """Complete at most 32 heterogeneous matched pairs in one bounded group.

    The science collector uses one original 48-game atomic block. The ordinary
    collector uses at most sixteen pairs, commits an ordered complete-unit
    prefix, and checks stop/budget conditions before each group. No rounds or
    half-pair results are durable here. This function does not authorize work.
    """
    requests=tuple(requests)
    if not requests or len(requests)>MAX_PAIRS or any(not isinstance(x,PairRequest) for x in requests):raise ValueError('one to thirty-two complete pair requests required')
    if len({x.root.rules.size for x in requests})!=1:raise ValueError('batched evaluation uses one board size')
    outputs=[];games=[];locations=[]
    for request in requests:
        outputs.append({'root_sha256':hashlib.sha256(request.root.canonical_json().encode()).hexdigest(),'subject':[],'incumbent':[]})
        for color,player in enumerate((BLACK,WHITE)):
            mine=seed_for(request.unit_seed,'subject',color);theirs=seed_for(request.unit_seed,'opponent',request.opponent.sha256,color)
            for key,subject in (('subject',request.subject),('incumbent',request.incumbent)):
                if subject is None:continue
                games.append(_Game(subject,request.opponent,request.root,player,mine,theirs));locations.append((len(outputs)-1,key))
    while True:
        active=[game for game in games if game.result is None]
        if not active:break
        actions=commit_evaluation_actions([(actor,game.state,player) for game in active
            for actor,player in ((game.subject,game.player),(game.opponent,3-game.player))])
        # Resolve only after every private commitment in this wave is fixed.
        for index,game in enumerate(active):game.advance(actions[2*index],actions[2*index+1])
    for game,(index,key) in zip(games,locations):outputs[index][key].append(game.result)
    for request,output in zip(requests,outputs):
        output['utility']=sum(g['subject_utility'] for g in output['subject'])/2
        output['incumbent_utility']=None if request.incumbent is None else sum(g['subject_utility'] for g in output['incumbent'])/2
        output['improvement']=None if request.incumbent is None else output['utility']-output['incumbent_utility']
    return outputs
