from pathlib import Path
import hashlib,json,os,subprocess,time
repo=Path('/home/gcao/proj/graygo');frozen=repo/'.worktrees/terminal-batch-f93f95b';root=Path(__file__).resolve().parent;start=time.monotonic();epoch=time.time()
def source():
 h=hashlib.sha256();p=frozen/'src/graygo'
 for f in sorted([*p.glob('*.py'),*(p/'native').glob('*.cpp')]):h.update(f.relative_to(p).as_posix().encode()+b'\0'+f.read_bytes()+b'\0')
 return h.hexdigest()
assert source()=='d2743ee50b4a5f28395670e545898ac324f0d68bf4d8d89147b7da1317d49546'
assert not subprocess.check_output(['git','-C',str(frozen),'status','--porcelain'],text=True)
body={'version':1,'status':'running','original_driver_pid':os.getpid(),'source_sha256':source(),'source_commit':subprocess.check_output(['git','-C',str(frozen),'rev-parse','HEAD'],text=True).strip(),'new_formal_A2_records':0,'allocation_interval_start_epoch':epoch,'physical_gpu_uuid':'GPU-446c23a3-a50d-fb18-6e54-d0b290eb78c2','time_ceiling_seconds':900,'scope':'whole package regression suite; tests/diagnostics only, no learning gate or price qualification'}
def save():(root/'job.json').write_text(json.dumps(body,indent=2,sort_keys=True)+'\n')
save();env=dict(os.environ,PYTHONPATH=str(frozen/'src')+os.pathsep+str(frozen)+os.pathsep+str(frozen/'tests'))
with (root/'tests.txt').open('w') as out:
 p=subprocess.Popen([str(repo/'.venv/bin/python'),'-m','unittest','discover','-s','tests','-v'],cwd=frozen,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);body['test_pid']=p.pid;save()
 try:code=p.wait(timeout=900)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=20);code=p.returncode;body['watchdog_exhausted']=True
body.update(status='passed' if code==0 else 'failed',returncode=code,source_unchanged=source()==body['source_sha256'],measured_driver_seconds=time.monotonic()-start,allocation_interval_end_epoch=time.time());save()
