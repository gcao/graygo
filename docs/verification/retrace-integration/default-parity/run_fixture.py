from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from graygo.manifest import source_digest
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.engine import Rules
out=Path(sys.argv[1]);out.mkdir()
records=0;payload={}
for task in ('delayed','branching','gray-mixed'):
 config=NFSPTaskConfig(task=task,learner='neural',purpose='profile',records=32 if task=='gray-mixed' else 80,
  eval_every=16,batch=4,learning_start=4,capacity=128,target_every=8,q_reuse=4,horizons=(5,),equivalent_actions=3,
  reservoir_open=0,handoff_end=8,epsilon_decay=40,hidden=8,width=8,blocks=1,norm_groups=2,time_cap_seconds=120)
 trainer=NFSPTaskTrainer(config,out/task,reserve_bytes=0);result=trainer.run(verbose=False)
 assert result['status']=='complete' and not result['formal_case_passed'] and not result['a2_complete']
 payload[task]={'state':trainer._state(),'replay':trainer.replay.arrays,'reservoir':trainer.reservoir.arrays,'metrics':trainer.metrics,'episode_bytes':(out/task/'episodes.bin').read_bytes()}
 records+=trainer.records
config=BoardRuntimeConfig(rules=Rules(3,5),envs=4,batch=8,learning_start=8,capacity=64,target_every=16,
 width=8,blocks=1,norm_groups=2,reservoir_open=0,handoff_end=8,q_reuse=4)
runtime=BoardNFSPRuntime(config);result=runtime.run_profile(64,time_cap_seconds=120)
assert result['status']=='complete' and not result['formal_learning_run']
payload['board']={'state':runtime.state_dict(),'replay':runtime.replay.arrays,'reservoir':runtime.reservoir.arrays}
records+=runtime.records
torch.save(payload,out/'mutable-state.pt')
(out/'summary.json').write_text(json.dumps({'source_sha256':source_digest(),'actual_diagnostic_records':records,
 'scientific_records':0,'cases':list(payload),'all_default_one_step':True},indent=2)+'\n')
