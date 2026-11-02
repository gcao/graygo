from pathlib import Path
import json,os,subprocess,time
from graygo.manifest import source_digest
root=Path(__file__).resolve().parent;started=time.monotonic();source=source_digest()
job={'version':1,'driver_pid':os.getpid(),'status':'running','source_sha256':source,'kind':'complete-suite-verification','formal_learning_run':False}
def save():(root/'job.json').write_text(json.dumps(job,indent=2)+'\n')
save()
with (root/'tests.txt').open('w') as output:
 p=subprocess.Popen(['/home/gcao/proj/graygo/.venv/bin/python','-m','unittest','discover','-s','tests','-v'],stdout=output,stderr=subprocess.STDOUT)
 job['test_pid']=p.pid;save();code=p.wait()
job.update(status='passed' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-started,source_unchanged=source_digest()==source);save()
