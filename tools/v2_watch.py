"""Standard-library observer for the concurrent v2 campaign."""
import json
from pathlib import Path
import time
try:
    from tools.training_benchmarks import evidence_stats
except ModuleNotFoundError:
    from training_benchmarks import evidence_stats


def read(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {}


def fields(root):
    root=Path(root);s=read(root/'status.json');state=read(root/'state.json').get('body',{})
    counters=s.get('counters',{});job=state.get('job');phase=s.get('phase','starting')
    try:
        process=Path('/proc')/str(s.get('pid'));alive=(process/'stat').read_text().split()[2]!='Z' and str(root).encode() in (process/'cmdline').read_bytes().split(b'\0')
    except OSError:alive=False
    health='RUNNING' if s.get('status')=='running' and alive else 'SUPERVISOR_EXITED' if s.get('status')=='running' else 'STOPPED'
    checkpoint=read(root/'resume/latest.json').get('generation','0')
    result=[f"state={health} status={s.get('status','unknown')} phase={phase}",
            'campaign=v2 rules=graygo/2 encoding=ko-hints/2',f"live_records={s.get('records',0):,} checkpoint_records={int(checkpoint):,}",
            f"Q_updates={counters.get('q_updates',0):,} avg_updates={counters.get('supervised_updates',0):,}",
            f"warm_start_records={s.get('parent_records','?')}",'replay=v2-only time_limit=none',
            f"native_decoder={s.get('native_decoder',False)} envs={s.get('envs','?')}",
            f"promotions={state.get('promotions',0)} champion={state.get('champion',{}).get('label','?')}"]
    if s.get('q_loss') is not None:result.append(f"Q_loss={s['q_loss']:.6f}")
    if s.get('mean_rounds') is not None:result.append(f"mean_rounds={s['mean_rounds']:.1f}")
    physical=s.get('physical_rounds',0)
    if physical:result.append(f"repetition_rejected={s.get('repetition_rejections',0)/physical:.1%}")
    result.append(f"evaluation=async learner_continues={health=='RUNNING'} gate_attempts={state.get('gate_attempts',0)}")
    if job:
        result.append(f"job={job['mode']} candidate_records={job['candidate']['records']:,}")
        folder=Path(job['root'])/'paired-betting-v4';progress=read(folder/'parallel-progress.json')
        counts=progress.get('games_per_pairing',{})
        if counts:result.append('job_progress='+','.join(f'{key}:{value}/{progress["look"]}' for key,value in counts.items()))
        spec=read(Path(job['root'])/'job.json').get('body',{})
        for label in ('head','historical-0','historical-1','historical-2'):
            stats=evidence_stats(folder/label/'evidence.sqlite')
            if stats:
                shown='champion' if label=='head' else label
                opponent=(spec.get('champion',{}) if label=='head' else spec.get('pool',{}).get(label,{})).get('label','?')
                result.append(f"average_vs_{shown}({opponent})={stats['wins']}W/{stats['draws']}D/{stats['losses']}L score={stats['win_score']:.1%} n={stats['games']}")
    last=state.get('last_gate')
    if last:
        d=last['decision'];failed=','.join(k for k,v in d['criteria'].items() if not v) or 'none'
        result.append(f"last_head_score={d['head_win_score']['mean']:.1%} lower={d['head_win_score']['lower']:.1%}")
        result.append(f"last_gate_records={last['records']:,} gate_pass={d['numeric_pass']} failed={failed} alpha={d['alpha_per_attempt']:.3g}")
    return result
