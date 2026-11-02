"""Commands for playing, reproducible baseline games, profiling and exact solving."""
import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from .engine import BLACK, WHITE, Rules, State, resolve, score
from .agents import uniform, tactical
from .manifest import make_manifest, write_report

AGENTS={"random":uniform,"tactical":tactical}


def render(state):
    glyph=".BW#"
    return "\n".join(" ".join(glyph[x] for x in state.board[r*state.rules.size:(r+1)*state.rules.size])
                     for r in range(state.rules.size))


def _rules(args):
    return Rules(args.size,args.limit,args.collision)


def simulate(args):
    rules=_rules(args)
    outcomes=[]; lengths=[]; collisions=0; hashes=set(); terminations={}
    started=time.perf_counter()
    for game in range(args.games):
        state=State(rules)
        black_rng=random.Random(args.seed+game*2)
        white_rng=random.Random(args.seed+game*2+1)
        transcript=hashlib.sha256()
        while not state.done:
            b=AGENTS[args.black](state,BLACK,black_rng)
            w=AGENTS[args.white](state,WHITE,white_rng)
            transcript.update(f"{b},{w};".encode())
            result=resolve(state,b,w)
            collisions+=result.collision is not None
            state=result.state
        hashes.add(transcript.hexdigest())
        lengths.append(state.rounds_played)
        outcomes.append(score(state.board,rules.size).utility_black)
        terminations[state.terminal_reason]=terminations.get(state.terminal_reason,0)+1
    return {"games":args.games,"black_wins":outcomes.count(1),"white_wins":outcomes.count(-1),
            "draws":outcomes.count(0),"mean_black_utility":statistics.mean(outcomes),
            "mean_rounds":statistics.mean(lengths),"min_rounds":min(lengths),"max_rounds":max(lengths),
            "collisions":collisions,"distinct_trajectories":len(hashes),"terminations":terminations,
            "elapsed_seconds":time.perf_counter()-started}


def profile(args):
    rules=_rules(args); state=State(rules); rng=random.Random(args.seed)
    games=0; began=time.perf_counter()
    for _ in range(args.rounds):
        state=resolve(state,rng.choice(state.legal_actions(BLACK)),rng.choice(state.legal_actions(WHITE))).state
        if state.done: games+=1;state=State(rules)
    seconds=time.perf_counter()-began
    return {"joint_rounds":args.rounds,"completed_games":games,"seconds":seconds,
            "rounds_per_second":args.rounds/seconds,"engine":"immutable_python_reference"}


def solve(args):
    from .solving import FiniteSolver
    if args.state:
        state=State.from_dict(json.loads(Path(args.state).read_text()))
    else:
        state=State(_rules(args))
    solver=FiniteSolver(max_nodes=args.max_nodes)
    value,b,w,solution=solver.solve(state)
    return {"state":state.to_dict(),"nodes":solver.nodes,"value_black":value,
            "black_actions":b,"white_actions":w,
            "black_policy":None if solution is None else solution.row.tolist(),
            "white_policy":None if solution is None else solution.column.tolist(),
            "numerical_gap":0 if solution is None else solution.residual_gap,
            "numerical_value_error_bound":solver.numerical_value_error_bound(state)}


def audit_native(args):
    from .audit import compare_native
    return compare_native(args.library,rounds=args.rounds,seed=args.seed)


def profile_network(args):
    import torch
    from .encoding import encode
    from .models import PolicyValueNet,masked_log_probs
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    model=PolicyValueNet().to(device)
    o=encode(State(_rules(args)),BLACK)
    x=torch.tensor(o.spatial[None],device=device).expand(args.batch,-1,-1,-1)
    g=torch.tensor(o.global_features[None],device=device).expand(args.batch,-1)
    mask=torch.tensor(o.legal[None],device=device).expand(args.batch,-1)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    def sync():
        if device.type=="cuda":torch.cuda.synchronize(device)
    for _ in range(3):
        with torch.no_grad():model(x,g)
    sync();t=time.perf_counter()
    for _ in range(args.batches):
        with torch.no_grad():model(x,g)
    sync();inference=time.perf_counter()-t
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    t=time.perf_counter()
    for _ in range(args.batches):
        logits,value=model(x,g)
        loss=-masked_log_probs(logits,mask)[:,0].mean()+value.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    sync();update=time.perf_counter()-t
    return {"device":str(device),"parameters":sum(p.numel() for p in model.parameters()),
            "batch":args.batch,"batches":args.batches,"inference_seconds":inference,
            "synthetic_update_seconds":update,"synthetic_only":True,
            "peak_cuda_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None}


def profile_buffers(args):
    from .buffer_profile import profile_buffers as benchmark
    return benchmark(_rules(args),args.buffer_dir,capacity=args.capacity,records=args.records,
                     iterations=args.iterations,batch=args.batch,seed=args.seed,storage=args.storage,native_library=args.native_library)


def profile_board(args):
    from .board_profile import profile_board as benchmark
    from .board_runtime import BoardRuntimeConfig
    config=BoardRuntimeConfig(rules=_rules(args),envs=args.envs,seed=args.seed,device=args.device,
                              q_reuse=args.q_reuse,native_library=args.native_library)
    return benchmark(config,args.run_dir,records=args.records,time_cap_seconds=args.time_cap_hours*3600,
                     stop_requested=_signal_flag())


def run_board_diagnostic(args,*,prior_setup_seconds=0.):
    started=time.monotonic()
    from .board_execution import BoardExecutor,BoardExecutionPlan
    from .board_journal import BoardJournalPlan
    from .board_runtime import BoardRuntimeConfig
    if _rules(args)!=Rules():raise ValueError('board execution diagnostic uses official 9x9/200-round gray rules')
    if type(args.records) is not int or args.records<2 or args.records%2 or args.records>65536:
        raise ValueError('execution diagnostic has an even total horizon at most 65536 records')
    if args.until_records is not None and (args.until_records<0 or args.until_records%2 or args.until_records>args.records):
        raise ValueError('paired operational endpoint within the unchanged diagnostic horizon required')
    config=BoardRuntimeConfig(seed=args.seed,envs=args.envs,device=args.device,q_reuse=args.q_reuse,
        native_library=args.native_library,reservoir_open=args.reservoir_open,handoff_end=args.handoff_end)
    points=tuple(sorted({0,args.records,*[n for n in (4096,args.reservoir_open,args.handoff_end) if n<=args.records]}))
    journal=BoardJournalPlan('diagnostic',args.records,points,args.evaluation_seed)
    execution=BoardExecutionPlan(args.time_cap_hours*3600,args.checkpoint_every,args.heartbeat_seconds)
    stopped=_signal_flag()
    with BoardExecutor(args.run_dir,config,journal,execution,resume=args.resume,
            prior_setup_seconds=prior_setup_seconds+time.monotonic()-started) as run:
        until=args.records if args.until_records is None else args.until_records
        boundary=min(until,args.reservoir_open) if args.opening_decision and run.runtime.records<=args.reservoir_open else until
        result=run.run(boundary,stop_requested=stopped)
        if (args.opening_decision and run.runtime.records==args.reservoir_open
                and result['status'] in ('opening_pending','checkpoint_boundary')):
            run.journal.diagnostic_opening(args.opening_decision)
            result=run.run(until,stop_requested=stopped)
    # Include the final, durably closed session clock in the command receipt.
    result.update(run.report(result['status']))
    return result


def generate_exposure(args):
    import fcntl
    from .curriculum import create_catalog,ExposureCatalog
    rules=_rules(args)
    if rules.size!=9 or rules.max_rounds!=200:raise ValueError('gray-exposure catalog requires 9x9/200-round rules')
    catalog=create_catalog(args.seed,train_families=args.train_families,heldout_families=args.heldout_families,max_attempts=args.max_attempts)
    path=Path(args.catalog).expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name('.'+path.name+'.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            if ExposureCatalog.from_dict(json.loads(path.read_text()))!=catalog:raise FileExistsError('frozen exposure catalog differs; use a new declared catalog')
        else:write_report(path,catalog.to_dict())
    return {'catalog_path':str(path),'catalog_sha256':catalog.sha256,'train_families':len(catalog.train),'heldout_families':len(catalog.heldout),
            'candidate_independent_generator':True,'collision_probe_families_certified':0,'training_started':False}


def build_probes(args):
    from .probe_runs import build_probes as construct
    if _rules(args)!=Rules():raise ValueError('probe generator v1 requires default 9x9/200-round rules')
    return construct(args.run_dir,seed=args.seed,attempt_cap=args.attempts,time_cap_seconds=args.time_cap_hours*3600,stop_requested=_signal_flag())


def verify_probes(args):
    from .probe_runs import verify_probes as verify
    if _rules(args)!=Rules():raise ValueError('probe benchmark v1 requires default 9x9/200-round rules')
    return verify(args.benchmark,regenerate=args.regenerate,time_cap_seconds=args.time_cap_hours*3600)


def science_evaluate(args):
    from .board_runtime import configure_runtime
    from .curriculum import ExposureCatalog
    from .evaluation import BaselinePolicy,FrozenNeuralPolicy
    from .science_protocol import ScienceProtocol,policy_keys
    from .science_evaluation import ScientificRun
    if _rules(args)!=Rules():raise ValueError('scientific contrast fixes both 9x9/200-round evaluation rules')
    started=time.monotonic();configure_runtime(args.device,True)
    import torch
    torch.set_num_threads(1)
    catalog=ExposureCatalog.from_dict(json.loads(Path(args.catalog).read_text()));policies={};paths={}
    if args.registry:
        registry=json.loads(Path(args.registry).read_text())
        if not isinstance(registry,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in registry.items()):raise ValueError('scientific registry must map curriculum/rule/seed labels to archive paths')
        bindings=[key+'='+value for key,value in registry.items()]
    else:bindings=args.policy
    for binding in bindings:
        if '=' not in binding:raise ValueError('policy bindings use curriculum/rule/seed=archive or baseline:name')
        label,value=binding.split('=',1);parts=label.split('/')
        if len(parts)!=3:raise ValueError('policy labels use curriculum/rule/seed')
        key=(parts[0],parts[1],int(parts[2]))
        if key in policies:raise ValueError('duplicate scientific policy binding')
        if value.startswith('baseline:'):
            if args.purpose!='diagnostic':raise ValueError('baselines cannot be relabeled as confirmation models')
            policies[key]=BaselinePolicy(value.split(':',1)[1])
        else:
            policies[key]=FrozenNeuralPolicy.from_archive(value,representation=args.representation,device=args.device,label=label);paths[key]=value
    if set(policies)!=set(policy_keys()):raise ValueError('all twelve curriculum/rule/seed policy bindings required')
    protocol=ScienceProtocol(args.name,catalog.sha256,{key:policy.sha256 for key,policy in policies.items()},args.time_cap_hours*3600,
        seed=args.seed,block_looks=tuple(args.block_looks),purpose=args.purpose,training_records=args.training_records,representation=args.representation)
    run=ScientificRun(protocol,args.run_dir,policies,catalog,archive_paths=paths,resume=args.resume,prior_setup_seconds=time.monotonic()-started)
    collection=run.run(args.games,stop_requested=_signal_flag());available=collection['committed_blocks']
    completed=[n for n in protocol.block_looks if n<=available]
    summary=None if not completed else run.estimate(max(completed)*48)
    if summary is not None:write_report(Path(args.run_dir)/'summary.json',summary)
    return {'collection':collection,'summary':summary,'gray_specialization_demonstrated':False}


def science_report(args):
    from .science_evaluation import ScientificEvidence
    if _rules(args)!=Rules():raise ValueError('scientific contrast fixes both 9x9/200-round evaluation rules')
    return ScientificEvidence(args.run_dir).estimate(args.games)


def train_matrix(args):
    from .matrix_training import MatrixPPOConfig,MatrixPPOTrainer
    config=MatrixPPOConfig(task=args.task,seed=args.seed,records=args.records,rollout=args.rollout,
                          eval_every=args.eval_every,entropy=args.entropy,device=args.device,
                          time_cap_seconds=args.time_cap_hours*3600)
    stop=_signal_flag()
    return MatrixPPOTrainer(config,args.run_dir,resume=args.resume).run(stop_requested=stop)


def preflight(args):
    from .preflight import run_ppo_preflight
    return run_ppo_preflight(args.run_dir,records=args.records,device=args.device,
                             time_cap_seconds=args.time_cap_hours*3600,stop_requested=_signal_flag())


def _signal_flag():
    import signal
    stopped=[False]
    def stop(signum,frame):stopped[0]=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    return lambda:stopped[0]


def play(args):
    state=State(_rules(args)); rng=random.Random(args.seed)
    while not state.done:
        # The AI commits using the unchanged public state before the human input.
        ai=AGENTS[args.opponent](state,WHITE,rng)
        print(render(state)); print(f"Round {state.rounds_played+1}; black forbidden: {sorted(state.forbidden_black)}")
        while True:
            try:
                text=input("Black move (row,col, 1-based; or pass): ").strip()
                if text=="pass": action=state.rules.pass_action
                else:
                    r,c=map(int,text.split(","));action=state.rules.action(r-1,c-1)
                if action not in state.legal_actions(BLACK): raise ValueError("occupied or forbidden")
                break
            except ValueError as exc: print(f"Invalid move: {exc}")
            except EOFError: return {"interrupted":True,"state":state.to_dict()}
        state=resolve(state,action,ai).state
        print(f"Revealed: black={action}, white={ai}")
    print(render(state)); result=score(state.board,state.rules.size)
    return {"black_score":str(result.black),"white_score":str(result.white),"reason":state.terminal_reason}


def train_nfsp(args):
    import signal
    from .nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
    stopped=[]
    previous={}
    for name in (signal.SIGINT,signal.SIGTERM):
        previous[name]=signal.signal(name,lambda signum,frame:stopped.append(signum))
    try:
        config=NFSPMatrixConfig(task=args.task,learner=args.learner,seed=args.seed,records=args.records,
            eval_every=args.eval_every,device=args.device,q_reuse=args.q_reuse,reservoir_open=args.reservoir_open,
            handoff_end=args.handoff_end,uniform_comparison=args.uniform_comparison,label_rule=args.label_rule,
            time_cap_seconds=args.time_cap_hours*3600)
        return NFSPMatrixTrainer(config,args.run_dir,resume=args.resume).run(stop_requested=lambda:bool(stopped))
    finally:
        for name,handler in previous.items():signal.signal(name,handler)


def train_nfsp_task(args):
    import hashlib
    import signal
    from .task_training import NFSPTaskConfig,NFSPTaskTrainer
    from .manifest import source_digest
    budget_hash=None
    if args.budget_profile:
        profile=Path(args.budget_profile).read_bytes();budget=json.loads(profile)
        if budget.get("source_sha256")!=source_digest() or budget.get("task")!=args.task or budget.get("learner")!=args.learner or budget.get("q_reuse")!=args.q_reuse or budget.get("storage","features")!=args.storage:
            raise ValueError("budget profile does not match this source/task/learner/reuse")
        if not budget.get("complete_pipeline") or set(budget.get("phases",[]))!={"pre","post"}:
            raise ValueError("budget profile must cover pre/post-opening complete learner work")
        from .native_decode import file_sha256
        actual_native=None if args.native_library is None else file_sha256(args.native_library)
        if budget.get("native_library_sha256")!=actual_native:raise ValueError("budget profile uses a different decoder binary")
        budget_hash=hashlib.sha256(profile).hexdigest()
    config=NFSPTaskConfig(time_cap_seconds=args.time_cap_hours*3600,task=args.task,learner=args.learner,seed=args.seed,
        records=args.records,eval_every=args.eval_every,device=args.device,q_reuse=args.q_reuse,
        reservoir_open=args.reservoir_open,handoff_end=args.handoff_end,uniform_comparison=args.uniform_comparison,
        horizons=tuple(args.horizons),equivalent_actions=args.equivalent_actions,hidden=args.hidden,
        width=args.width,blocks=args.blocks,norm_groups=args.norm_groups,eval_batch=args.eval_batch,
        capacity=args.capacity,purpose=args.purpose,budget_profile_sha256=budget_hash,storage=args.storage,native_library=args.native_library)
    stopped=[];previous={}
    for name in (signal.SIGINT,signal.SIGTERM):previous[name]=signal.signal(name,lambda signum,frame:stopped.append(signum))
    try:return NFSPTaskTrainer(config,args.run_dir,resume=args.resume).run(stop_requested=lambda:bool(stopped))
    finally:
        for name,handler in previous.items():signal.signal(name,handler)


def build_native(args):
    from .native_decode import build_native as build
    return build(args.output_dir,compiler=args.compiler)


def evaluate(args):
    from .confidence import ConfidenceFamily
    from .evaluation import BaselinePolicy,FrozenNeuralPolicy,EvaluationPlan,EvaluationRun
    data=json.loads(Path(args.confidence_plan).read_text());family=ConfidenceFamily(**data)
    def policy(name,archive,representation,label):
        if name in ('random','tactical'):
            if archive is not None:raise ValueError("baseline does not use a checkpoint")
            return BaselinePolicy(name)
        if archive is None:raise ValueError("neural evaluation requires an archived public checkpoint")
        return FrozenNeuralPolicy.from_archive(archive,representation=representation,device=args.device,label=label)
    subject=policy(args.subject,args.subject_archive,args.representation,args.subject_label)
    opponent=policy(args.opponent,args.opponent_archive,args.opponent_representation,'opponent')
    incumbent=None if args.incumbent is None else policy(args.incumbent,args.incumbent_archive,args.incumbent_representation,'incumbent')
    plan=EvaluationPlan(args.name,args.subject_label,args.endpoint,_rules(args),args.seed,family.looks,confidence_sha256=family.sha256,comparison='improvement' if incumbent is not None else 'utility')
    run=EvaluationRun(plan,args.run_dir,subject,opponent,incumbent=incumbent,resume=args.resume)
    stopped=_signal_flag();collection=run.run(args.games,time_cap_seconds=args.time_cap_hours*3600,stop_requested=stopped)
    estimate=None
    if collection['committed_pairs']>=args.games//2:estimate=run.estimate(family,args.games,comparison='improvement' if incumbent is not None else 'utility')
    return {'collection':collection,'confidence':estimate,'promotion_permitted':False,'scope':'fixed paired evaluation; promotion requires calibrated protected-pool protocol and A2/smoke evidence'}


def calibrate_promotion(args):
    import hashlib
    from .promotion import PromotionProtocol,PairedEvidence
    from .confidence import canonical_sha
    from .promotion_calibration import PilotPopulation,calibrate
    from .manifest import source_digest
    protocol=PromotionProtocol.from_dict(json.loads(Path(args.protocol).read_text()))
    paths={}
    for binding in args.pilot:
        if "=" not in binding:raise ValueError("pilot bindings use label=directory")
        label,path=binding.split("=",1)
        if label in paths:raise ValueError("duplicate pilot opponent binding")
        paths[label]=path
    evidence={label:PairedEvidence.load(path) for label,path in paths.items()}
    pilot=PilotPopulation.from_evidence(protocol,evidence)
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=True)
    plan={"protocol":protocol.to_dict(),"protocol_sha256":protocol.sha256,"pilot_sha256":pilot.sha256,
          "pilot_evidence":pilot.evidence,"simulations":args.simulations,"seed":args.seed,"source_sha256":source_digest()}
    import fcntl
    path=root/"plan.json";plan['time_ceiling_seconds']=args.time_cap_hours*3600
    with (root/".calibration.lock").open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if json.loads(path.read_text())!=plan:raise ValueError("calibration plan/source/pilot changed")
            saved=json.loads((root/"progress.json").read_text())
            if saved.get("format_version")!=1 or saved.get("sha256")!=canonical_sha(saved.get("state")):raise ValueError("calibration progress checksum differs")
            progress=saved["state"]
        else:
            if path.exists():raise FileExistsError("calibration exists; inspect/resume its original process")
            write_report(path,plan);progress=None
        def persist(value):write_report(root/"progress.json",{"format_version":1,"state":value,"sha256":canonical_sha(value)})
        stopped=_signal_flag()
        report,state=calibrate(protocol,pilot,simulations=args.simulations,seed=args.seed,time_cap_seconds=args.time_cap_hours*3600,
            progress=progress,save_progress=persist,stop_requested=stopped)
        persist(state);write_report(root/"calibration.json",report)
        return report


def main(argv=None):
    command_started=time.monotonic()
    parser=argparse.ArgumentParser(prog="graygo")
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("simulate","profile","solve","play","audit-native","profile-network","train-matrix","preflight","profile-buffers","train-nfsp","train-nfsp-task","build-native","evaluate","calibrate-promotion","profile-board","run-board-diagnostic","generate-exposure","build-probes","verify-probes","science-evaluate","science-report"):
        p=commands.add_parser(name)
        p.add_argument("--size",type=int,default=9)
        p.add_argument("--limit",type=int)
        p.add_argument("--collision",choices=("gray","remove"),default="gray")
        p.add_argument("--seed",type=int,default=0)
        p.add_argument("--output")
        if name=="calibrate-promotion":
            p.add_argument("--protocol",required=True);p.add_argument("--pilot",action="append",required=True)
            p.add_argument("--run-dir",required=True);p.add_argument("--simulations",type=int,default=2000)
            p.add_argument("--time-cap-hours",type=float,required=True);p.add_argument("--resume",action="store_true")
        elif name=="evaluate":
            p.add_argument("--run-dir",required=True);p.add_argument("--confidence-plan",required=True)
            p.add_argument("--name",required=True);p.add_argument("--subject-label",required=True);p.add_argument("--endpoint",required=True)
            p.add_argument("--games",type=int,default=500);p.add_argument("--time-cap-hours",type=float,required=True)
            p.add_argument("--subject",choices=("random","tactical","neural"),default="random")
            p.add_argument("--subject-archive");p.add_argument("--representation",choices=("average","greedy_q"),default="average")
            p.add_argument("--opponent",choices=("random","tactical","neural"),default="tactical")
            p.add_argument("--opponent-archive");p.add_argument("--opponent-representation",choices=("average","greedy_q"),default="average")
            p.add_argument("--incumbent",choices=("random","tactical","neural"));p.add_argument("--incumbent-archive")
            p.add_argument("--incumbent-representation",choices=("average","greedy_q"),default="average")
            p.add_argument("--device",default="cpu");p.add_argument("--resume",action="store_true")
        elif name=="simulate":
            p.add_argument("--games",type=int,default=10)
            p.add_argument("--black",choices=AGENTS,default="random")
            p.add_argument("--white",choices=AGENTS,default="random")
        elif name=="profile":p.add_argument("--rounds",type=int,default=1000)
        elif name=="solve":
            p.add_argument("--state");p.add_argument("--max-nodes",type=int,default=100000)
        elif name=="audit-native":
            p.add_argument("--library",required=True);p.add_argument("--rounds",type=int,default=3000)
        elif name=="build-native":
            p.add_argument("--output-dir",required=True);p.add_argument("--compiler",default="c++")
        elif name=="profile-network":
            p.add_argument("--device",default="cpu");p.add_argument("--batch",type=int,default=16)
            p.add_argument("--batches",type=int,default=10)
        elif name=="profile-buffers":
            p.add_argument("--native-library")
            p.add_argument("--storage",choices=("features","rule-state"),default="features")
            p.add_argument("--buffer-dir",required=True);p.add_argument("--capacity",type=int,default=500000)
            p.add_argument("--records",type=int,default=10000);p.add_argument("--iterations",type=int,default=50)
            p.add_argument("--batch",type=int,default=512)
        elif name=="science-evaluate":
            p.add_argument("--run-dir",required=True);p.add_argument("--catalog",required=True);p.add_argument("--name",required=True)
            registry=p.add_mutually_exclusive_group(required=True)
            registry.add_argument("--policy",action="append");registry.add_argument("--registry")
            p.add_argument("--games",type=int,default=24000)
            p.add_argument("--time-cap-hours",type=float,required=True);p.add_argument("--device",default="cpu")
            p.add_argument("--purpose",choices=("confirmation","diagnostic"),default="confirmation")
            p.add_argument("--block-looks",nargs="+",type=int,default=[500,2000]);p.add_argument("--training-records",type=int,default=80000000)
            p.add_argument("--representation",choices=("average","actor"),default="average");p.add_argument("--resume",action="store_true")
        elif name=="science-report":
            p.add_argument("--run-dir",required=True);p.add_argument("--games",type=int,required=True)
        elif name=="build-probes":
            p.set_defaults(size=9,limit=200,collision="gray")
            p.add_argument("--run-dir",required=True);p.add_argument("--attempts",type=int,default=10000)
            p.add_argument("--time-cap-hours",type=float,required=True)
        elif name=="verify-probes":
            p.set_defaults(size=9,limit=200,collision="gray")
            p.add_argument("--benchmark",required=True);p.add_argument("--regenerate",action="store_true")
            p.add_argument("--time-cap-hours",type=float,default=.05)
        elif name=="generate-exposure":
            p.add_argument("--catalog",required=True);p.add_argument("--train-families",type=int,default=64)
            p.add_argument("--heldout-families",type=int,default=32);p.add_argument("--max-attempts",type=int,default=100000)
        elif name=="run-board-diagnostic":
            p.add_argument("--run-dir",required=True);p.add_argument("--records",type=int,default=16384)
            p.add_argument("--until-records",type=int);p.add_argument("--time-cap-hours",type=float,required=True)
            p.add_argument("--device",default="cpu");p.add_argument("--envs",type=int,default=16)
            p.add_argument("--q-reuse",type=int,choices=(1,4),default=1);p.add_argument("--native-library")
            p.add_argument("--reservoir-open",type=int,default=8192);p.add_argument("--handoff-end",type=int,default=12288)
            p.add_argument("--evaluation-seed",type=int,default=41);p.add_argument("--checkpoint-every",type=int,default=4096)
            p.add_argument("--heartbeat-seconds",type=float,default=30.)
            p.add_argument("--opening-decision",choices=("pass","no_go"));p.add_argument("--resume",action="store_true")
        elif name=="profile-board":
            p.add_argument("--run-dir",required=True);p.add_argument("--records",type=int,default=16384)
            p.add_argument("--time-cap-hours",type=float,required=True);p.add_argument("--device",default="cpu")
            p.add_argument("--envs",type=int,default=16);p.add_argument("--q-reuse",type=int,choices=(1,4),default=1)
            p.add_argument("--native-library")
        elif name=="train-nfsp-task":
            p.add_argument("--native-library")
            p.add_argument("--storage",choices=("features","rule-state"),default="features")
            p.add_argument("--run-dir",required=True);p.add_argument("--task",choices=("delayed","delayed-forced","branching","gray-pure","gray-mixed"),default="delayed")
            p.add_argument("--learner",choices=("neural","tabular"),default="neural")
            p.add_argument("--records",type=int,default=15000000);p.add_argument("--eval-every",type=int,default=500000)
            p.add_argument("--device",default="cpu");p.add_argument("--q-reuse",type=int,choices=(1,4),default=1)
            p.add_argument("--time-cap-hours",type=float,required=True);p.add_argument("--purpose",choices=("a2","profile"),default="a2")
            p.add_argument("--budget-profile");p.add_argument("--resume",action="store_true")
            p.add_argument("--reservoir-open",type=int,default=5000000);p.add_argument("--handoff-end",type=int,default=7000000)
            p.add_argument("--uniform-comparison",action="store_true");p.add_argument("--capacity",type=int,default=500000)
            p.add_argument("--horizons",nargs="+",type=int,default=[50,100,200]);p.add_argument("--equivalent-actions",type=int,default=40)
            p.add_argument("--hidden",type=int,default=64);p.add_argument("--width",type=int,default=64)
            p.add_argument("--blocks",type=int,default=8);p.add_argument("--norm-groups",type=int,default=8);p.add_argument("--eval-batch",type=int,default=128)
        elif name=="train-nfsp":
            p.add_argument("--run-dir",required=True);p.add_argument("--task",choices=("asymmetric","rps","matching-pennies","pure82"),default="asymmetric")
            p.add_argument("--learner",choices=("neural","tabular"),default="neural")
            p.add_argument("--records",type=int,default=15000000);p.add_argument("--eval-every",type=int,default=500000)
            p.add_argument("--device",default="cpu");p.add_argument("--q-reuse",type=int,choices=(1,4),default=1)
            p.add_argument("--reservoir-open",type=int,default=5000000);p.add_argument("--handoff-end",type=int,default=7000000)
            p.add_argument("--time-cap-hours",type=float,default=8);p.add_argument("--resume",action="store_true")
            p.add_argument("--uniform-comparison",action="store_true");p.add_argument("--label-rule",choices=("greedy","executed"),default="greedy")
        elif name in ("train-matrix","preflight"):
            p.add_argument("--run-dir",required=True)
            p.add_argument("--records",type=int,default=2000000)
            p.add_argument("--device",default="cpu")
            p.add_argument("--time-cap-hours",type=float,default=24)
            if name=="train-matrix":
                p.add_argument("--task",choices=("asymmetric","rps","matching-pennies","pure82"),default="asymmetric")
                p.add_argument("--entropy",choices=("taper","fixed-0.1"),default="taper")
                p.add_argument("--rollout",type=int,default=1024)
                p.add_argument("--eval-every",type=int,default=102400)
                p.add_argument("--resume",action="store_true")
        else:p.add_argument("--opponent",choices=AGENTS,default="tactical")
    args=parser.parse_args(argv)
    if any(getattr(args,name,1)<1 for name in ("games","rounds","batch","batches")):
        parser.error("games and rounds must be positive")
    try:
        matrix_command=args.command in ("train-matrix","preflight","train-nfsp","train-nfsp-task")
        rules=None if matrix_command or args.command in ("build-native","calibrate-promotion") else _rules(args)
        metadata=make_manifest(seed=args.seed,command=vars(args),rules=None if rules is None else
                               {"size":rules.size,"max_rounds":rules.max_rounds,"collision":rules.collision})
        if args.command in ("build-native","calibrate-promotion"):metadata.update({"rule_version":None,"baseline_version":None,"encoding_version":None})
        if matrix_command:
            metadata.update({"rule_version":None,"baseline_version":None,"encoding_version":"graygo/matrix-role/1"})
            if args.command=="preflight":metadata.update({"seed":None,"seeds":[0,1,2]})
            if args.command=="train-nfsp":metadata["encoding_version"]="graygo/nfsp-matrix-role/1"
        if args.command=="run-board-diagnostic":
            result=run_board_diagnostic(args,prior_setup_seconds=time.monotonic()-command_started)
        else:result=globals()[args.command.replace("-","_")](args)
        if args.command=="solve":metadata["rules"]=result["state"]["rules"]
        if args.command=="train-nfsp-task":
            actual=json.loads((Path(args.run_dir)/"run.json").read_text())
            metadata.update({key:actual[key] for key in ("rules","rule_version","encoding_version","baseline_version")})
        if args.command in ("science-evaluate","science-report"):
            metadata["rules"]={"size":9,"max_rounds":200,"collision":"gray/remove"}
        report={"manifest":metadata,"result":result}
        if args.output:write_report(args.output,report)
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(1,f"graygo: {exc}\n")
