"""Managed, unlimited champion training across recoverable archive segments."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from graygo.confidence import canonical_sha
from graygo.evaluation import seed_for
from graygo.manifest import source_digest,write_report
from graygo.storage import DEFAULT_RESERVE,digest
from tools.learning_loop import prepare,worker


def stage_alpha(index):
    # Telescoping allocation: sum over all future segments is exactly .05.
    if type(index) is not int or index<0:raise ValueError('nonnegative segment index required')
    return .05/((index+1)*(index+2))


def next_endpoint(records,segment_records):
    return (records//segment_records+1)*segment_records


def create(root,parent,parent_checkout,publication,*,project=None,segment_records=20000000,block=262144,reserve_bytes=DEFAULT_RESERVE):
    if any(type(n) is not int or n<2 or n%2 for n in (segment_records,block)):
        raise ValueError('paired segment and checkpoint sizes required')
    root=Path(root).resolve();checkout=Path(__file__).resolve().parents[1]
    body={'version':'graygo/continuous-training/1','parent_run':str(Path(parent).resolve()),
          'parent_checkout':str(Path(parent_checkout).resolve()),'publication':str(Path(publication).resolve()),
          'project':str(Path(project or checkout).resolve()),'segment_records':segment_records,'block':block,
          'checkout':str(checkout),'source_sha256':source_digest(),'controller_sha256':digest(Path(__file__)),
          'time_limit_seconds':None,'record_limit':None,'gate_total_alpha':.05,
          'scope':'owner-authorized experimental continuous learning; formal scientific grants remain separate'}
    root.mkdir(parents=True,exist_ok=False)
    write_report(root/'plan.json',{'body':body,'sha256':canonical_sha(body)},reserve_bytes=reserve_bytes)
    state={'plan_sha256':canonical_sha(body),'completed_segments':0,'current_stage':None,
           'parent_run':body['parent_run'],'parent_checkout':body['parent_checkout']}
    write_report(root/'state.json',state,reserve_bytes=reserve_bytes)
    write_report(root/'status.json',{'status':'prepared','unlimited':True},reserve_bytes=reserve_bytes)
    return body


def run(root,*,max_segments=None,reserve_bytes=DEFAULT_RESERVE):
    root=Path(root).resolve();lock=(root/'.controller.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=[];previous={kind:signal.getsignal(kind) for kind in (signal.SIGTERM,signal.SIGINT)}
    for kind in previous:signal.signal(kind,lambda *unused:stopped.append(True))
    try:
        wrapper=json.loads((root/'plan.json').read_text());plan=wrapper['body']
        if (wrapper['sha256']!=canonical_sha(plan) or plan['source_sha256']!=source_digest()
                or plan['controller_sha256']!=digest(Path(__file__))):raise ValueError('continuous training plan/source changed')
        state=json.loads((root/'state.json').read_text())
        if state['plan_sha256']!=wrapper['sha256']:raise ValueError('controller state belongs to a different plan')
        completed_here=0
        while not stopped and (max_segments is None or completed_here<max_segments):
            index=state['completed_segments']
            if state['current_stage'] is None:
                parent=Path(state['parent_run'])
                clock=json.loads((parent/'execution.json').read_text())['body']['clock']
                endpoint=next_endpoint(clock['last_checkpoint']['records'],plan['segment_records'])
                stage=root/'stages'/f'{index:06d}'
                if not (stage/'plan.json').exists():
                    prepare(stage,parent,state['parent_checkout'],plan['publication'],total_seconds=None,
                        records=endpoint,block=plan['block'],gate_alpha=stage_alpha(index),
                        gate_seed=seed_for(73,'continuous-segment',index),reserve_bytes=reserve_bytes)
                body=json.loads((stage/'plan.json').read_text())['body']
                if (body['remaining_seconds'] is not None or body['records']!=endpoint
                    or body.get('gate_alpha')!=stage_alpha(index)
                    or body.get('gate_seed')!=seed_for(73,'continuous-segment',index)
                    or body['initialization']['parent_run']!=str(parent)):
                    raise ValueError('recovered segment differs from its fixed continuation plan')
                state['current_stage']=str(stage)
                write_report(root/'state.json',state,reserve_bytes=reserve_bytes)
            stage=Path(state['current_stage']);body=json.loads((stage/'plan.json').read_text())['body']
            receipt={'run_dir':str(stage),'campaign_dir':str(root),'process_root':str(root),
                'supervisor_pid':os.getpid(),'started_at':time.time(),'source_checkout':plan['checkout'],
                'parent_records':body['initialization']['records'],'publication':plan['publication'],
                'gatekeeping_enabled':True,'arena_publication':True,'unlimited':True,'deadline_utc':None,
                'remaining_seconds':None,'segment':index,'archive_segment_end':body['records']}
            write_report(stage/'launch-receipt.json',receipt,reserve_bytes=reserve_bytes)
            write_report(Path(plan['project'])/'runs/latest-training-loop.json',receipt,reserve_bytes=reserve_bytes)
            write_report(root/'status.json',{'status':'running','unlimited':True,'current_stage':str(stage),
                'completed_segments':index,'record_limit':None,'time_limit_seconds':None},reserve_bytes=reserve_bytes)
            if stopped:break
            result=worker(stage,stop_state=stopped,reserve_bytes=reserve_bytes)
            if result['status']=='complete' and result['total_records']==body['records']:
                state.update(completed_segments=index+1,current_stage=None,parent_run=str(stage/'learner'),
                             parent_checkout=plan['checkout'])
                write_report(root/'state.json',state,reserve_bytes=reserve_bytes)
                completed_here+=1
                continue
            write_report(root/'status.json',{'status':'stopped','unlimited':True,'reason':result['status'],
                'current_stage':str(stage),'records':result['total_records']},reserve_bytes=reserve_bytes)
            return result
        write_report(root/'status.json',{'status':'stopped' if stopped else 'segment_boundary','unlimited':True,
            'completed_segments':state['completed_segments'],'current_stage':state['current_stage']},reserve_bytes=reserve_bytes)
        return state
    except BaseException as error:
        write_report(root/'status.json',{'status':'failed','unlimited':True,'error':repr(error)},reserve_bytes=reserve_bytes)
        raise
    finally:
        for kind,handler in previous.items():signal.signal(kind,handler)
        lock.close()


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--run-dir',required=True);parser.add_argument('--parent-run');parser.add_argument('--parent-checkout')
    parser.add_argument('--publication');parser.add_argument('--project');args=parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    if args.prepare:
        if None in (args.parent_run,args.parent_checkout,args.publication):parser.error('parent checkpoint/source and publication required')
        result=create(args.run_dir,args.parent_run,args.parent_checkout,args.publication,project=args.project)
    else:result=run(args.run_dir)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
