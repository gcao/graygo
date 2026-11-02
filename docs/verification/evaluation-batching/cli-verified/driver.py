from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from graygo.manifest import source_digest
from graygo.evaluation_evidence import FrozenEvaluationEvidence
repo=Path('/home/gcao/proj/graygo');frozen=repo/'.worktrees/evaluation-batching-a05655e';root=Path(__file__).resolve().parent
archive=repo/'runs/probe-policy-cli-zero-pathfix/policies/200/weights/000000000000'
expected_source='f4c8b427d823d204df6bd46371b38b35bcb776d783ca04addd6831f84e666cbd'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree(path):return {str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
assert source_digest()==expected_source
confidence=root/'confidence.json';confidence.write_text(json.dumps({'name':'batch-cli-diagnostic','endpoints':['utility'],'looks':[4,8],'alpha':.05})+'\n')
inputs=[archive/'manifest.json',archive/'models.pt',archive.parents[1]/'run.json',confidence];before={str(p):sha(p) for p in inputs}
job={'version':1,'status':'running','original_driver_pid':os.getpid(),'source_commit':'a05655e','source_sha256':expected_source,'maximum_driver_seconds':180,
 'purpose':'bounded record-zero CLI/SQLite-prefix/changed-mode check; no training or price qualification','children':[],'formal_learning_run':False,'execution_authorized':False}
started=time.monotonic();job['allocation_interval_start_epoch']=time.time()
def save():(root/'job.json').write_text(json.dumps(job,indent=2,sort_keys=True)+'\n')
save()
common=[str(repo/'.venv/bin/python'),'-m','graygo','evaluate','--run-dir',str(root/'evidence'),'--confidence-plan',str(confidence),'--name','batch-cli-diagnostic',
 '--subject-label','record-zero-smoke','--endpoint','utility','--subject','neural','--subject-archive',str(archive),'--representation','average',
 '--opponent','tactical','--device','cuda','--inference-batch-size','32','--pair-batch-size','2','--time-cap-hours','.02']
def invoke(label,extra,expected=0,changed=False):
 cmd=list(common)
 if changed:cmd[cmd.index('--inference-batch-size')+1]='16'
 cmd+=extra+['--output',str(root/(label+'.json'))]
 with (root/(label+'.stdout')).open('w') as out,(root/(label+'.stderr')).open('w') as err:
  process=subprocess.Popen(cmd,cwd=frozen,stdout=out,stderr=err)
  job['children'].append({'pid':process.pid,'label':label,'command':cmd});save()
  try:code=process.wait(timeout=min(60,max(1,180-(time.monotonic()-started))))
  except BaseException:
   process.kill();process.wait();raise
 assert code==expected,(label,code,(root/(label+'.stderr')).read_text())
try:
 invoke('first-look',['--games','4']);first=FrozenEvaluationEvidence.load(root/'evidence');assert len(first.rows)==2
 first_rows=first.rows
 invoke('resumed-look',['--games','8','--resume']);second=FrozenEvaluationEvidence.load(root/'evidence')
 assert len(second.rows)==4 and second.rows[:2]==first_rows
 body=json.loads((root/'resumed-look.json').read_text());result=body['result'];policy=second.contract['subject']
 assert result['collection']['status']=='complete' and second.plan.pair_batch_size==2 and policy['inference']['batch_size']==32
 assert policy['origin']['records']==0 and not result['promotion_permitted']
 evidence_before=tree(root/'evidence');invoke('changed-mode',['--games','8','--resume'],expected=1,changed=True)
 assert 'differ' in (root/'changed-mode.stderr').read_text() and tree(root/'evidence')==evidence_before
 assert source_digest()==expected_source and before=={str(p):sha(p) for p in inputs}
 assert body['manifest']['git_commit'].startswith('a05655e') and not body['manifest']['git_dirty']
 job.update(status='verified_diagnostic',physical_games=8,committed_pairs=4,prefix_identical_on_resume=True,changed_mode_rejected_without_input_writes=True,
  original_archive_and_confidence_unchanged=True,source_unchanged=True,record_zero_weights_not_learning_evidence=True,read_only_evidence_sha256=second.sha256)
except BaseException as exc:job.update(status='failed',error=repr(exc))
finally:
 job.update(elapsed_seconds=time.monotonic()-started,allocation_interval_end_epoch=time.time());save()
