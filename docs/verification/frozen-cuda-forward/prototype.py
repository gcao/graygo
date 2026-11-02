from copy import deepcopy
from pathlib import Path
import json,os,time,torch
from graygo.board_runtime import configure_runtime
from graygo.models import QNet,PolicyNet
from graygo.manifest import source_digest
root=Path(__file__).resolve().parent;started=time.monotonic();configure_runtime('cuda',True);torch.set_num_threads(1)
results=[]
def sync():torch.cuda.synchronize()
def timed(fn):
 sync();at=time.perf_counter();out=fn();sync();return out,time.perf_counter()-at
with torch.no_grad():
 for kind in (QNet,PolicyNet):
  models=[]
  for seed in (41,42,43):
   torch.manual_seed(seed);models.append(kind().to('cuda').eval().requires_grad_(False))
  for batch in (1,2,32):
   x=torch.randn(batch,30,9,9,device='cuda');g=torch.randn(batch,14,device='cuda')
   template=deepcopy(models[0]);layout=[];cursor=0
   for name,param in template.named_parameters():
    cursor=(cursor+63)//64*64;layout.append((name,cursor,param.shape,param.numel()));cursor+=param.numel()
   storage=torch.zeros((cursor+63)//64*64,device='cuda')
   for name,offset,shape,count in layout:dict(template.named_parameters())[name].data=storage[offset:offset+count].view(shape)
   packed=[]
   for model in models:
    value=torch.zeros_like(storage)
    for name,offset,shape,count in layout:value[offset:offset+count].copy_(dict(model.named_parameters())[name].flatten())
    packed.append(value)
   sx=x.clone();sg=g.clone();storage.copy_(packed[0]);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   at=time.perf_counter()
   with torch.cuda.stream(stream):
    for _ in range(3):template(sx,sg)
   torch.cuda.current_stream().wait_stream(stream);sync();graph=torch.cuda.CUDAGraph()
   cpu_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state().clone()
   with torch.cuda.graph(graph):output=template(sx,sg)
   sync();capture_seconds=time.perf_counter()-at
   assert torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(cuda_before,torch.cuda.get_rng_state())
   maxdiff=0.;exact=True;eager_seconds=[];graph_seconds=[]
   for index,model in enumerate(models):
    for trial in range(4):
     xx=x+trial/10;gg=g-trial/10
     expected=model(xx,gg)
     storage.copy_(packed[index]);sx.copy_(xx);sg.copy_(gg);graph.replay();actual=output.clone()
     maxdiff=max(maxdiff,float((actual-expected).abs().max()));exact=exact and bool(torch.equal(actual,expected))
    for _ in range(16):
     _,seconds=timed(lambda:model(x,g));eager_seconds.append(seconds)
     def replay():
      storage.copy_(packed[index]);sx.copy_(x);sg.copy_(g);graph.replay();return output.clone()
     _,seconds=timed(replay);graph_seconds.append(seconds)
   results.append({'model':kind.__name__,'batch':batch,'weights':len(models),'comparisons':12,'exact':exact,'max_absolute_difference':maxdiff,
    'capture_seconds_includes_three_warmups':capture_seconds,'eager_forward_seconds':eager_seconds,'shared_graph_copy_and_forward_seconds':graph_seconds})
report={'version':1,'status':'complete','pid':os.getpid(),'source_sha256':source_digest(),'elapsed_seconds':time.monotonic()-started,
 'torch_version':str(torch.__version__),'results':results,'new_learning_records':0,'scope':'bounded shared static-weight CUDA forward prototype; synthetic full-size networks/inputs, no actor sampling or campaign price'}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'seconds':report['elapsed_seconds'],'exact':[r['exact'] for r in results]},indent=2))
