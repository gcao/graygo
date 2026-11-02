from pathlib import Path
import json,os,subprocess,time
from graygo.manifest import source_digest
root=Path(__file__).resolve().parent;started=time.monotonic();source=source_digest()
job={'version':1,'driver_pid':os.getpid(),'status':'running','source_sha256':source,'kind':'complete-suite-verification','formal_learning_run':False}
(root/'job.json').write_text(json.dumps(job,indent=2)+'\n')
with (root/'tests.txt').open('w') as output:
 process=subprocess.Popen(['/home/gcao/proj/graygo/.venv/bin/python','-m','unittest','discover','-s','tests','-v'],stdout=output,stderr=subprocess.STDOUT)
 job['test_pid']=process.pid;(root/'job.json').write_text(json.dumps(job,indent=2)+'\n');code=process.wait()
job.update(status='passed' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-started,source_unchanged=source_digest()==source)
(root/'job.json').write_text(json.dumps(job,indent=2)+'\n')
