"""Complete bounded A2 game/observation domains under one pinned package.

No trainer, replay buffer, optimizer or scored game is created. All legal edges
are enumerated; limits fail the probe rather than truncate its domain.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from tools.task_actor_boundary import Transcript, fingerprint, file_sha, require

VERSION='graygo/task-domain-boundary/1'
CASES=tuple((task,learner,seed) for task in ('gray-pure','gray-mixed') for learner in ('neural','tabular') for seed in (0,1,2))


def complete_domain(game,roots,encode_state,*,max_nodes,max_edges,check_time=lambda:None):
    require(max_nodes>0 and max_edges>0,'positive complete-domain limits required')
    pending=deque(roots);states={};objects={};encoded={};edges=[];nonterminal=set();terminals=set()
    def key(state):
        data=encode_state(state);raw=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)
        identity=hashlib.sha256(raw.encode()).hexdigest()
        if identity in encoded:require(encoded[identity]==raw,'encoded-state digest collision')
        encoded[identity]=raw
        return identity
    root_keys=[key(root) for root in roots]
    while pending:
        check_time();state=pending.popleft();identity=key(state)
        if identity in states:
            require(objects[identity]==state,'state serialization aliases two public states')
            continue
        require(len(states)<max_nodes,'complete bounded domain exceeds its node budget')
        terminal=bool(game.terminal(state));row={'state':encode_state(state),'terminal':terminal}
        states[identity]=row;objects[identity]=state
        if terminal:
            value=float(game.utility(state));require(np.isfinite(value) and -1<=value<=1,'invalid bounded terminal utility')
            row['utility_black']=value;terminals.add(identity);continue
        nonterminal.add(identity);black=tuple(game.actions(state,1));white=tuple(game.actions(state,2))
        require(black and white and len(set(black))==len(black) and len(set(white))==len(white),'complete distinct legal action sets required')
        row['black_actions']=list(black);row['white_actions']=list(white)
        for b in black:
            for w in white:
                if len(edges)%64==0:check_time()
                require(len(edges)<max_edges,'complete bounded domain exceeds its edge budget')
                child=game.transition(state,b,w);target=key(child);edges.append((identity,int(b),int(w),target));pending.append(child)
    require(terminals and nonterminal and all(target in states for _,_,_,target in edges),'incomplete bounded game graph')
    return {'roots':root_keys,'states':dict(sorted(states.items())),'edges':sorted(edges),
            'nonterminal_ids':sorted(nonterminal),'terminal_ids':sorted(terminals)}


def probe(*,expected_source,time_cap_seconds):
    from graygo.manifest import source_digest
    from graygo.task_adapter import TaskAdapter
    from graygo.task_training import NFSPTaskConfig
    import scipy
    started=time.monotonic()
    require(source_digest()==expected_source,'pinned bounded-domain package differs')
    require(type(time_cap_seconds) in (float,int) and 0<time_cap_seconds<=600,'bounded positive domain time required')
    def check_time():require(time.monotonic()-started<=time_cap_seconds,'bounded-domain time cap exhausted')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);np.random.seed(1729)
    results={}
    for task,learner,seed in CASES:
        check_time()
        config=NFSPTaskConfig(time_cap_seconds=600,task=task,learner=learner,seed=seed,records=65536,eval_every=32768,q_reuse=4,purpose='profile',device='cpu')
        adapter=TaskAdapter(config)
        graph=complete_domain(adapter.game,adapter.roots,adapter.encode_state,max_nodes=config.max_nodes,max_edges=1000000,check_time=check_time)
        known={hashlib.sha256(json.dumps(adapter.encode_state(s),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() for s in adapter.states}
        require(known==set(graph['nonterminal_ids']),'adapter omits or adds a nonterminal public state')
        observations=Transcript();count=0
        for state in adapter.states:
            for role in (1,2):
                obs=adapter.game.observation(state,role)
                require(tuple(np.flatnonzero(obs.legal))==tuple(adapter.game.actions(state,role)),'observation mask differs from complete legal actions')
                observations.add((adapter.encode_state(state),role,obs.spatial,obs.global_features,obs.legal));count+=1
        torch.manual_seed(100000+seed);q,average=adapter.models();q.eval().requires_grad_(False);average.eval().requires_grad_(False)
        initial_rng=torch.get_rng_state().clone();forwards=Transcript()
        with torch.no_grad():
            for index,obs in enumerate(adapter.observations):
                if index%32==0:check_time()
                inputs={'spatial':torch.tensor(obs.spatial[None],dtype=torch.float32),'globals':torch.tensor(obs.global_features[None],dtype=torch.float32)}
                forwards.add((index,q(**inputs),average(**inputs)))
        require(torch.equal(initial_rng,torch.get_rng_state()),'bounded-domain forward consumed a global RNG')
        name=f'{task}/{learner}/{seed}'
        results[name]={'config':config.plan(),'complete_domain_sha256':fingerprint(graph),'adapter_sha256':fingerprint(adapter.manifest()),
            'observations_sha256':observations.sha256,'initial_models_sha256':fingerprint({'q':q.state_dict(),'average':average.state_dict()}),
            'raw_forwards_sha256':forwards.sha256,'torch_rng_sha256':fingerprint(initial_rng),
            'counts':{'states':len(graph['states']),'nonterminal':len(graph['nonterminal_ids']),'terminal':len(graph['terminal_ids']),
                      'joint_transitions':len(graph['edges']),'observations':count,'raw_forwards':2*len(adapter.observations)},
            'root_values':adapter.values,'complete':True}
        del q,average,adapter,graph
    require(source_digest()==expected_source,'bounded-domain package changed during probe');check_time()
    return {'version':VERSION,'status':'complete','source_sha256':expected_source,'cases':results,
        'runtime':{'device':'cpu','threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),
                   'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),'scipy':scipy.__version__},
        'source_unchanged':True,'elapsed_seconds':time.monotonic()-started,'new_training_records':0,'new_scored_evaluation_games':0,
        'execution_authorized':False,'scope':'Complete fixed bounded A2 game graphs, both-role observations and common-model seeded raw forwards for every original seed/learner. No unrestricted-board, training-convergence or project-entry claim.'}


def verify_pair(reference,candidate,*,reference_source,candidate_source):
    expected={f'{task}/{learner}/{seed}' for task,learner,seed in CASES}
    for value,source in ((reference,reference_source),(candidate,candidate_source)):
        require(value['version']==VERSION and value['status']=='complete' and value['source_sha256']==source and value['source_unchanged'] is True,'incomplete or changed domain probe')
        require(set(value['cases'])==expected,'complete original bounded family/learner/seed inventory required')
        require(value['runtime']['device']=='cpu' and value['runtime']['threads']==1 and value['runtime']['deterministic'] is True,'domain execution mode differs')
        require(value['new_training_records']==value['new_scored_evaluation_games']==0 and value['execution_authorized'] is False,'domain probe became learning authority')
        for name,row in value['cases'].items():
            task,learner,seed=name.split('/');config=row['config']
            require(config['task']==task and config['learner']==learner and config['seed']==int(seed)
                    and config['purpose']=='profile' and config['device']=='cpu' and config['storage']=='features'
                    and config['q_reuse']==4 and (config['width'],config['blocks'],config['norm_groups'])==(64,8,8),'bounded family/learner/seed or shape binding differs')
            c=row['counts'];require(row['complete'] is True and c['states']==c['terminal']+c['nonterminal'] and c['terminal']>0 and c['nonterminal']>0
                and c['joint_transitions']>0 and c['observations']==2*c['nonterminal'] and c['raw_forwards']==2*c['observations'],'partial domain/observation inventory')
    require(reference['runtime']==candidate['runtime'],'domain numerical runtime differs')
    require(reference['cases']==candidate['cases'],'bounded domain, observations, model outputs or fixed configuration differ')
    return {'version':VERSION,'status':'complete_bounded_A2_domains_identical','reference_source_sha256':reference_source,
        'candidate_source_sha256':candidate_source,'counts':{name:row['counts'] for name,row in reference['cases'].items()},
        'case_count':len(expected),'execution_authorized':False,'a2_evidence_reusable':False,
        'scope':'Complete finite domain equality for both fixed bounded families, both learners, three original codebook seeds and the recorded numerical runtime. Separate default-collector, update, original-cohort and resource checks remain required.'}


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--expected-source-sha',required=True);parser.add_argument('--time-cap-seconds',type=float,default=600);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    try:
        out=Path(args.output).resolve();require(not out.exists(),'preserve original domain result; choose a new output')
        result=probe(expected_source=args.expected_source_sha,time_cap_seconds=args.time_cap_seconds);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as file:json.dump(result,file,indent=2,sort_keys=True,allow_nan=False);file.write('\n')
        print(json.dumps({'status':result['status'],'cases':len(result['cases']),'elapsed_seconds':result['elapsed_seconds']},indent=2));return 0
    except (ValueError,RuntimeError,OSError,KeyError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
