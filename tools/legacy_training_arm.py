"""Bounded search-training reference using frozen graygo-old libraries.

Never imports run.py or its notifications/promotion gate. Each completed chunk
saves a recoverable model, optimizer, replay and RNG checkpoint atomically.
The parent comparison watchdog owns the hard process-group deadline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',required=True);parser.add_argument('--root',required=True)
    parser.add_argument('--seconds',required=True,type=float);parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--games',type=int,default=2)
    parser.add_argument('--visits',type=int,default=900);parser.add_argument('--board-size',type=int,default=9)
    parser.add_argument('--max-turns',type=int,default=100);parser.add_argument('--blocks',type=int,default=12)
    parser.add_argument('--filters',type=int,default=128);parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--max-chunks',type=int)
    args=parser.parse_args()
    if not 0<args.seconds<=3600 or min(args.games,args.visits,args.batch)<1:parser.error('bounded positive training configuration required')
    started=time.monotonic();stopped=[]
    for kind in (signal.SIGTERM,signal.SIGINT):signal.signal(kind,lambda *unused:stopped.append(True))
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'latest.pt').exists():raise FileExistsError('comparison arms never reset or implicitly resume an existing learner')
    sys.path.insert(0,str(Path(args.repo).resolve()))
    # The explicit imports exclude the old launcher and notification code.
    from model import GrayGoNet
    from train import TrainConfig,ReplayBuffer,create_optimizer,train_model
    from selfplay import SelfPlayConfig
    from selfplay_cpp import generate_selfplay_data_cpp
    from export_model import export
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    model=GrayGoNet(board_size=args.board_size,blocks=args.blocks,filters=args.filters).to(args.device)
    cfg=TrainConfig(batch_size=args.batch,train_epochs=2.,learning_rate=3e-4,rng_seed=args.seed)
    optimizer=create_optimizer(model,cfg);replay=ReplayBuffer(maxlen=cfg.replay_buffer_size)
    chunks=records=updates=0
    shape={'board_size':args.board_size,'blocks':args.blocks,'filters':args.filters}
    def save():
        blob={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'replay_buffer':replay.state_dict(),
              'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
              'metadata':{'config':shape,'chunks':chunks,'fresh_records':records,'optimizer_batches':updates,
                          'elapsed_seconds':time.monotonic()-started,'initialization':'fresh random; no teacher checkpoint',
                          'training_rules':'graygo-old native rules; max_turns='+str(args.max_turns)}}
        # Keep Python replay objects in a private bundle. The public weights
        # remain loadable through the existing strict weights_only adapter.
        for name,payload in [('resume.pt',blob),('latest.pt',{'state_dict':blob['state_dict'],'metadata':blob['metadata']})]:
            if name=='latest.pt':
                with (root/'resume.pt').open('rb') as stream:payload['metadata']['resume_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
            temporary=root/('.'+name+'.tmp')
            with temporary.open('wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
            temporary.replace(root/name)
        descriptor=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    save()
    while not stopped and time.monotonic()-started<args.seconds and (args.max_chunks is None or chunks<args.max_chunks):
        # Export uses CPU copies and random verification inputs; preserve learner RNG.
        devices=[torch.cuda.current_device()] if args.device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):export(model,root/'model_traced.pt')
        sp=SelfPlayConfig(board_size=args.board_size,games_per_iteration=args.games,max_turns=args.max_turns,
                          num_visits=args.visits,rng_seed=args.seed+100*chunks)
        samples=generate_selfplay_data_cpp(root/'model_traced.pt',sp,device=args.device)
        # A completed collection chunk is durable even if time expires before fitting.
        replay.extend(samples);records+=len(samples);save()
        if stopped or time.monotonic()-started>=args.seconds:break
        cfg.rng_seed=args.seed+100*chunks+11
        metrics=train_model(model,replay,cfg,optimizer=optimizer)
        updates+=metrics['num_batches'];chunks+=1;save()
        print(json.dumps({'chunks':chunks,'fresh_records':records,'optimizer_batches':updates,'metrics':metrics}),flush=True)
    print(json.dumps({'status':'stopped' if stopped else 'complete','chunks':chunks,'fresh_records':records,
                      'optimizer_batches':updates,'elapsed_seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
