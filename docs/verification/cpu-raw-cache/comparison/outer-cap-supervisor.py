from pathlib import Path
import hashlib,json,os,signal,time
root=Path(__file__).resolve().parent
expected={'pid': 2779568, 'start_ticks': '56658474', 'cmdline_sha256': '65691efc838a36cb34740b9a420b1fa1c78aa5d8f9199cf020ae66686202e828', 'outer_cap_seconds': 780, 'original_loaded_alarm_seconds': 960, 'children': {'gray-mixed': 2779577, 'gray-pure': 2779576}}
body={'version':1,'original_parent':expected,'original_supervisor_pid':os.getpid(),'status':'enforcing_original_declared_outer_cap','no_training_restart':True,'new_control_or_gpu_allocation':False}
def save():
 p=root/'outer-cap-supervisor.tmp';p.write_text(json.dumps(body,indent=2,sort_keys=True)+'\n');p.replace(root/'outer-cap-supervisor.json')
def alive():
 try:
  stat=Path('/proc/'+str(expected['pid'])+'/stat').read_text().rsplit(')',1)[1].split();command=Path('/proc/'+str(expected['pid'])+'/cmdline').read_bytes()
 except FileNotFoundError:return False
 if stat[0] in ('Z','X'):return False
 assert stat[19]==expected['start_ticks'] and hashlib.sha256(command).hexdigest()==expected['cmdline_sha256'];return True
start=float(expected['start_ticks'])/os.sysconf('SC_CLK_TCK');deadline=start+expected['outer_cap_seconds'];save()
while alive():
 now=time.clock_gettime(time.CLOCK_BOOTTIME)
 if now>=deadline-30 and not body.get('declared_budget_stop_requested'):
  os.kill(expected['pid'],signal.SIGALRM);body['declared_budget_stop_requested']=True;save()
 if now>=deadline-1:
  job=json.loads((root/'job.json').read_text());pids=list(expected['children'].values())
  if job.get('original_comparison_pid'):pids.append(job['original_comparison_pid'])
  for pid in pids:
   try:
    stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
    if int(stat[1])==expected['pid']:os.killpg(pid,signal.SIGKILL)
   except ProcessLookupError:pass
   except FileNotFoundError:pass
  os.killpg(expected['pid'],signal.SIGKILL);body['hard_declared_cap_stop_requested']=True;save();break
 time.sleep(.5)
body['status']='original_parent_terminal_under_declared_supervision';body['observed_elapsed_from_parent_start_seconds']=time.clock_gettime(time.CLOCK_BOOTTIME)-start;save()
