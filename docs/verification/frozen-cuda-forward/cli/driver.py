from pathlib import Path
import hashlib,json,os,signal,subprocess,time
repo=Path('/home/gcao/proj/graygo');frozen=repo/'.worktrees/frozen-cuda-c3b8b5f';root=Path(__file__).resolve().parent
allocation=frozen/'docs/experiments/frozen-cuda-runtime-diagnostic-allocation.json';a=json.loads(allocation.read_text());started=time.monotonic();epoch=time.time();process=None
python=str(repo/'.venv/bin/python');source=a['source_sha256'];env=dict(os.environ,PYTHONPATH=str(frozen/'src'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree(path):return {str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
library=repo/a['native_library'];assert sha(library)==a['native_library_sha256']
job={'version':1,'status':'running','original_driver_pid':os.getpid(),'source_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),
 'source_sha256':source,'allocation_sha256':sha(allocation),'driver_sha256':sha(Path(__file__)),'commands':[],'outer_watchdog_seconds':600,
 'physical_gpu_uuid':a['physical_gpu_uuid'],'allocation_interval_start_epoch':epoch,'new_formal_learning_records':0,'new_control_records':0,'execution_authorized':False}
def save():
 p=root/'job.json';t=root/'job.tmp';t.write_text(json.dumps(job,indent=2,sort_keys=True)+'\n');t.replace(p)
def stop(*args):raise TimeoutError('original graph CLI diagnostic watchdog exhausted')
signal.signal(signal.SIGALRM,stop);signal.alarm(570);save()
def run(label,args,code=0):
 global process
 command=[python,'-m','graygo',*args,'--output',str(root/(label+'.json'))];entry={'label':label,'command':command};job['commands'].append(entry);save();at=time.monotonic()
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
  process=subprocess.Popen(command,cwd=frozen,env=env,stdout=out,stderr=err,start_new_session=True);entry['pid']=process.pid;save();result=process.wait(timeout=min(200,max(1,570-(time.monotonic()-started))))
 entry.update(returncode=result,seconds=time.monotonic()-at);save()
 assert result==code,(label,result,(root/(label+'.stderr')).read_text()[-3000:])
 return None if code else json.loads((root/(label+'.json')).read_text())
try:
 common=['run-board-diagnostic','--records','16384','--time-cap-hours','.05','--device','cuda','--envs','16','--q-reuse','4','--native-library',str(library),
  '--reservoir-open','4096','--handoff-end','8192','--checkpoint-every','4096','--heartbeat-seconds','30','--opening-decision','pass']
 eager=run('eager',common+['--run-dir',str(root/'eager')])
 split=run('graph-split',common+['--run-dir',str(root/'graph'),'--cuda-graphs','--until-records','8192'])
 final=run('graph-resumed',common+['--run-dir',str(root/'graph'),'--cuda-graphs','--resume'])
 before=tree(root/'graph');run('changed-runtime-mode',common+['--run-dir',str(root/'graph'),'--resume'],code=1)
 assert tree(root/'graph')==before
 archive=repo/'runs/probe-policy-cli-zero-pathfix/policies/200/weights/000000000000';archive_before=tree(archive)
 confidence=root/'confidence.json';confidence.write_text(json.dumps({'name':'graph-evaluation-diagnostic','endpoints':['utility'],'looks':[4,8],'alpha':.05})+'\n')
 evaluate=['evaluate','--run-dir',str(root/'evaluation'),'--confidence-plan',str(confidence),'--name','graph-evaluation-diagnostic','--subject-label','record-zero',
  '--endpoint','utility','--subject','neural','--subject-archive',str(archive),'--representation','average','--opponent','tactical','--device','cuda',
  '--inference-batch-size','32','--pair-batch-size','2','--time-cap-hours','.02']
 run('evaluation-first',evaluate+['--games','4','--cuda-graphs']);evaluated=run('evaluation-resumed',evaluate+['--games','8','--cuda-graphs','--resume'])
 before=tree(root/'evaluation');run('changed-evaluation-mode',evaluate+['--games','8','--resume'],code=1);assert tree(root/'evaluation')==before and tree(archive)==archive_before
 import sys
 sys.path.insert(0,str(frozen/'src'))
 import torch,numpy as np
 from graygo.storage import CheckpointStore
 from graygo.buffers import ReplayBuffer,ReservoirBuffer
 from graygo.evaluation_evidence import FrozenEvaluationEvidence
 from graygo.manifest import source_digest
 assert source_digest()==source
 def bundle(path):
  candidates=[p.parent for p in path.rglob('runtime.pt') if p.parent.name.isdigit()]
  result=max(candidates,key=lambda p:(int(p.name),p.parent.name=='final'));CheckpointStore.verify(result);return result
 left,right=bundle(root/'eager'),bundle(root/'graph');assert left.name==right.name=='000000016384'
 x=torch.load(left/'runtime.pt',map_location='cpu',weights_only=True);y=torch.load(right/'runtime.pt',map_location='cpu',weights_only=True)
 def equal(a,b):
  if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
  if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
  if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(c,d) for c,d in zip(a,b))
  return a==b
 assert equal(x['backend'],y['backend']) and equal(x['rng'],y['rng']) and equal(x['torch_rng'],y['torch_rng']) and equal(x['cuda_rng'],y['cuda_rng'])
 cx,cy=x['collector'],y['collector']
 for snapshot in cy['snapshots'].values():assert snapshot.pop('frozen_execution')['backend']=='shared-static-weight-cuda-graph'
 assert equal(cx,cy)
 for loader,name in [(ReplayBuffer,'replay.npz'),(ReservoirBuffer,'reservoir.npz')]:
  l,r=loader.load(left/name),loader.load(right/name)
  assert l.size==r.size and all(np.array_equal(v,r.arrays[k]) for k,v in l.arrays.items())
 evidence=FrozenEvaluationEvidence.load(root/'evaluation');assert evidence.available_games==8
 assert evidence.contract['subject']['origin']['records']==0 and evidence.contract['subject']['frozen_execution']['backend']=='shared-static-weight-cuda-graph'
 assert evaluated['manifest']['git_dirty'] is False and evaluated['manifest']['git_commit']==job['source_commit']
 job.update(status='verified_diagnostic',fresh_diagnostic_records=32768,matched_final_record=16384,optimizer_model_buffer_actor_and_rng_states_exact=True,
  runtime_mode_change_rejected_without_input_writes=True,evaluation_mode_change_rejected_without_input_writes=True,
  evaluation_games=8,record_zero_evaluation_not_strength_evidence=True,source_unchanged=True,eager_final_bundle=str(left),graph_final_bundle=str(right),
  source_runtime_caches_not_serialized=True,original_public_archive_unchanged=True)
except BaseException as exc:job.update(status='failed',error=repr(exc))
finally:
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
 job.update(measured_driver_seconds=time.monotonic()-started,allocation_interval_end_epoch=time.time());save();signal.alarm(0)
