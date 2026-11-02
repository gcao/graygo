"""Compact bounded replay/reservoir storage shared by board and task adapters.

Binary spatial features and legal masks are bit-packed; public scalar features
remain float32. Save/load uses NumPy arrays with allow_pickle=False. Callers must
quiesce writers before checkpoint serialization and checkpoint their sampler RNG.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np
from .encoding import Observation, ENCODING_VERSION

FORMAT_VERSION=1
DEFAULT_MAX_BYTES=2*1024**3


@dataclass(frozen=True)
class ObservationSpec:
    spatial_shape: tuple[int,...]
    globals: int
    actions: int
    version: str

    def __post_init__(self):
        object.__setattr__(self,"spatial_shape",tuple(self.spatial_shape))
        if not self.spatial_shape or any(type(n) is not int or n<1 for n in self.spatial_shape):
            raise ValueError("positive spatial dimensions required")
        if type(self.globals) is not int or self.globals<0 or type(self.actions) is not int or not 1<=self.actions<=65536:
            raise ValueError("invalid scalar/action dimensions")
        if not isinstance(self.version,str) or not self.version:
            raise ValueError("encoding version required")

    @classmethod
    def board(cls,size=9):
        if type(size) is not int or size<3:raise ValueError("invalid board size")
        return cls((30,size,size),14,size*size+1,ENCODING_VERSION)

    @property
    def spatial_bytes(self):return (math.prod(self.spatial_shape)+7)//8
    @property
    def legal_bytes(self):return (self.actions+7)//8

    def pack(self,observation):
        x,g,legal=map(np.asarray,(observation.spatial,observation.global_features,observation.legal))
        if x.shape!=self.spatial_shape or g.shape!=(self.globals,) or legal.shape!=(self.actions,):
            raise ValueError("observation dimensions disagree with buffer schema")
        if not np.isin(x,[0,1]).all() or not np.isfinite(g).all() or legal.dtype!=np.bool_:
            raise ValueError("binary features, finite globals, and boolean mask required")
        with np.errstate(over="ignore"):
            scalars=g.astype(np.float32,copy=True)
        if not np.isfinite(scalars).all():raise ValueError("globals overflow float32 storage")
        return (np.packbits(x.reshape(-1).astype(np.uint8),bitorder="little"),
                scalars,np.packbits(legal,bitorder="little"))

    def unpack(self,spatial,globals,legal):
        leading=spatial.shape[:-1]
        x=np.unpackbits(spatial,axis=-1,count=math.prod(self.spatial_shape),bitorder="little").reshape(*leading,*self.spatial_shape)
        mask=np.unpackbits(legal,axis=-1,count=self.actions,bitorder="little").astype(bool)
        return x,globals.copy(),mask


def _integer(value,name,*,maximum=2**64-1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 0<=value<=maximum:
        raise ValueError(f"invalid {name}")
    return int(value)


def _allocate(capacity,layout,max_bytes):
    if type(capacity) is not int or capacity<1:raise ValueError("positive capacity required")
    total=sum(capacity*math.prod(shape)*np.dtype(dtype).itemsize for dtype,shape in layout.values())
    if total>max_bytes:raise ValueError(f"buffer requires {total} bytes, above configured {max_bytes}")
    return {name:np.zeros((capacity,*shape),dtype=dtype) for name,(dtype,shape) in layout.items()}


def _save(path,metadata,arrays,size):
    with Path(path).open("wb") as file:
        np.savez_compressed(file,metadata=np.array(json.dumps(metadata,sort_keys=True)),
                            **{name:array[:size] for name,array in arrays.items()})


def _read(path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data["metadata"].item()))
        arrays={name:data[name].copy() for name in data.files if name!="metadata"}
    if metadata.get("format_version")!=FORMAT_VERSION:raise ValueError("unsupported buffer format")
    return metadata,arrays


def _restore_arrays(destination,loaded,size):
    if set(destination)!=set(loaded):raise ValueError("buffer array inventory mismatch")
    for name,stored in loaded.items():
        expected=destination[name]
        if stored.shape!=(size,*expected.shape[1:]) or stored.dtype!=expected.dtype:
            raise ValueError(f"invalid buffer array {name}")
        expected[:size]=stored


class ReplayBuffer:
    def __init__(self,capacity,spec,*,max_bytes=DEFAULT_MAX_BYTES):
        self.capacity,self.spec=capacity,spec
        layout={"spatial":(np.uint8,(spec.spatial_bytes,)),"globals":(np.float32,(spec.globals,)),
                "legal":(np.uint8,(spec.legal_bytes,)),"next_spatial":(np.uint8,(spec.spatial_bytes,)),
                "next_globals":(np.float32,(spec.globals,)),"next_legal":(np.uint8,(spec.legal_bytes,)),
                "action":(np.uint16,()),"reward":(np.float32,()),"terminated":(np.bool_,()),
                "behavior_prob":(np.float32,()),"stream":(np.uint64,()),"step":(np.uint32,()),
                "opponent_version":(np.uint64,()),"policy_version":(np.uint64,()),"mode":(np.uint8,()),
                "actor_mask":(np.bool_,()),"value_mask":(np.bool_,()),"supplemental":(np.bool_,()),"ordinal":(np.uint64,()),
                "next_index":(np.int64,()),"next_ordinal":(np.uint64,())}
        self.arrays=_allocate(capacity,layout,max_bytes)
        self.arrays["next_index"].fill(-1)
        self.size=self.cursor=self.insertions=0
        self._tails={}

    @property
    def allocated_bytes(self):return sum(a.nbytes for a in self.arrays.values())
    @property
    def snapshot_raw_bytes(self):return sum(a[:self.size].nbytes for a in self.arrays.values())+65536

    def add(self,observation,action,reward,next_observation,terminated,*,behavior_prob,
            stream,step,opponent_version=0,policy_version=0,mode=0,actor_mask=True,value_mask=True,supplemental=False):
        action=_integer(action,"action",maximum=self.spec.actions-1)
        stream=_integer(stream,"stream");step=_integer(step,"step",maximum=2**32-1)
        opponent_version=_integer(opponent_version,"opponent version");policy_version=_integer(policy_version,"policy version")
        mode=_integer(mode,"mode",maximum=1)
        if not all(isinstance(v,(bool,np.bool_)) for v in (terminated,actor_mask,value_mask,supplemental)):
            raise ValueError("terminal/eligibility flags must be bool")
        if supplemental and actor_mask:raise ValueError("supplemental coverage cannot be actor eligible")
        if not np.isfinite(reward) or not -1<=reward<=1 or not np.isfinite(behavior_prob) or not 0<behavior_prob<=1:
            raise ValueError("bounded reward and positive behavior probability required")
        if np.float32(behavior_prob)<=0:raise ValueError("behavior probability underflows float32 storage")
        if self.insertions>=2**64-1:raise OverflowError("replay ordinal exhausted")
        packed=self.spec.pack(observation);following=self.spec.pack(next_observation)
        if not observation.legal[action]:raise ValueError("replay source action was illegal")
        if bool(next_observation.legal.any())==bool(terminated):
            raise ValueError("next observation terminal mask disagrees with done")
        a=self.arrays;index=self.cursor
        previous=self._tails.get(stream)
        link=False
        if previous is not None and previous[0]!=index:
            slot,ordinal=previous
            if int(a["ordinal"][slot])==ordinal and not a["terminated"][slot] and int(a["step"][slot])+1==step and int(a["opponent_version"][slot])==opponent_version:
                if not all(np.array_equal(a["next_"+name][slot],data) for name,data in zip(("spatial","globals","legal"),packed)):
                    raise ValueError("consecutive replay records disagree on the shared state")
                link=True
        if self.size==self.capacity:
            old_stream=int(a["stream"][index]);old_ordinal=int(a["ordinal"][index])
            if self._tails.get(old_stream)==(index,old_ordinal):self._tails.pop(old_stream)
        for name,data in zip(("spatial","globals","legal"),packed):a[name][index]=data
        for name,data in zip(("next_spatial","next_globals","next_legal"),following):a[name][index]=data
        for name,value in (("action",action),("reward",reward),("terminated",terminated),("behavior_prob",behavior_prob),
                           ("stream",stream),("step",step),("opponent_version",opponent_version),("policy_version",policy_version),
                           ("mode",mode),("actor_mask",actor_mask),("value_mask",value_mask),("supplemental",supplemental),("ordinal",self.insertions)):
            a[name][index]=value
        a["next_index"][index]=-1;a["next_ordinal"][index]=0
        if link:
            a["next_index"][previous[0]]=index;a["next_ordinal"][previous[0]]=self.insertions
        if terminated:self._tails.pop(stream,None)
        else:self._tails[stream]=(index,self.insertions)
        self.insertions+=1;self.size=min(self.size+1,self.capacity);self.cursor=self.insertions%self.capacity
        return index

    def sample(self,count,rng):
        data=self.sample_sequences(count,1,rng)
        return {name:array[:,0] for name,array in data.items()}

    def sample_sequences(self,count,length,rng):
        if type(count) is not int or count<1 or type(length) is not int or length<1 or not self.size:
            raise ValueError("positive sample dimensions and nonempty replay required")
        indices=np.full((count,length),-1,dtype=np.int64)
        valid=np.zeros((count,length),dtype=bool)
        starts=rng.integers(0,self.size,count)
        a=self.arrays
        for row,start in enumerate(starts):
            index=int(start)
            for col in range(length):
                indices[row,col]=index;valid[row,col]=True
                if a["terminated"][index]:break
                following=int(a["next_index"][index])
                if following<0 or int(a["ordinal"][following])!=int(a["next_ordinal"][index]):break
                index=following
        safe=np.maximum(indices,0)
        data={name:array[safe].copy() for name,array in a.items() if name not in ("next_index","next_ordinal")}
        for name,array in data.items():array[~valid]=False if array.dtype==bool else 0
        data["terminated"][~valid]=True;data["behavior_prob"][~valid]=1
        for prefix in ("","next_"):
            spatial,globals,legal=self.spec.unpack(data.pop(prefix+"spatial"),data.pop(prefix+"globals"),data.pop(prefix+"legal"))
            data[prefix+"spatial"],data[prefix+"globals"],data[prefix+"legal"]=spatial,globals,legal
        data.update(indices=indices,valid=valid)
        return data

    def save(self,path):
        _save(path,{"format_version":FORMAT_VERSION,"kind":"replay","capacity":self.capacity,
                    "spec":asdict(self.spec),"size":self.size,"cursor":self.cursor,"insertions":self.insertions},self.arrays,self.size)

    @classmethod
    def load(cls,path,*,max_bytes=DEFAULT_MAX_BYTES):
        meta,arrays=_read(path)
        if meta.get("kind")!="replay":raise ValueError("not a replay snapshot")
        spec=ObservationSpec(**meta["spec"]);result=cls(meta["capacity"],spec,max_bytes=max_bytes)
        size=_integer(meta["size"],"size",maximum=result.capacity)
        insertions=_integer(meta["insertions"],"insertions")
        if size!=min(insertions,result.capacity) or meta["cursor"]!=insertions%result.capacity:
            raise ValueError("invalid replay cursor/counts")
        _restore_arrays(result.arrays,arrays,size)
        result.size,result.insertions,result.cursor=size,insertions,meta["cursor"]
        a=result.arrays
        if size and set(map(int,a["ordinal"][:size]))!=set(range(insertions-size,insertions)):
            raise ValueError("invalid live replay generations")
        if size:
            legal=np.unpackbits(a["legal"][:size],axis=-1,count=spec.actions,bitorder="little").astype(bool)
            next_legal=np.unpackbits(a["next_legal"][:size],axis=-1,count=spec.actions,bitorder="little").astype(bool)
            if (a["action"][:size]>=spec.actions).any() or not legal[np.arange(size),a["action"][:size]].all():
                raise ValueError("illegal restored source action")
            if (next_legal.any(-1)==a["terminated"][:size]).any():raise ValueError("restored done/mask mismatch")
            if not all(np.isfinite(a[name][:size]).all() for name in ("globals","next_globals","reward","behavior_prob")):
                raise ValueError("nonfinite replay state")
            if (np.abs(a["reward"][:size])>1).any() or (a["behavior_prob"][:size]<=0).any() or (a["behavior_prob"][:size]>1).any():
                raise ValueError("invalid replay reward/behavior probability")
            if (a["mode"][:size]>1).any() or (a["supplemental"][:size]&a["actor_mask"][:size]).any():
                raise ValueError("invalid replay eligibility")
        newest={}
        for index in range(size):
            ordinal=int(a["ordinal"][index]);stream=int(a["stream"][index])
            if ordinal%result.capacity!=index:raise ValueError("replay generation/slot mismatch")
            if stream not in newest or ordinal>newest[stream][1]:newest[stream]=(index,ordinal)
            following=int(a["next_index"][index])
            if following < -1 or following>=result.capacity:raise ValueError("invalid replay link")
            if following>=0 and int(a["ordinal"][following])==int(a["next_ordinal"][index]):
                if int(a["ordinal"][following])<=ordinal or a["stream"][following]!=a["stream"][index] or a["step"][following]!=a["step"][index]+1 or a["opponent_version"][following]!=a["opponent_version"][index]:
                    raise ValueError("replay link crosses a trajectory boundary")
        result._tails={stream:tail for stream,tail in newest.items() if not a["terminated"][tail[0]]}
        return result


class ReservoirBuffer:
    """Uniform Algorithm-R reservoir of eligible frozen greedy labels only."""
    def __init__(self,capacity,spec,*,opening_record=0,max_bytes=DEFAULT_MAX_BYTES):
        self.capacity,self.spec=capacity,spec
        self.opening_record=_integer(opening_record,"reservoir opening")
        layout={"spatial":(np.uint8,(spec.spatial_bytes,)),"globals":(np.float32,(spec.globals,)),
                "legal":(np.uint8,(spec.legal_bytes,)),"greedy":(np.uint8,(spec.legal_bytes,)),
                "record":(np.uint64,()),"policy_version":(np.uint64,())}
        self.arrays=_allocate(capacity,layout,max_bytes);self.size=self.seen=self.revisions=0

    @property
    def allocated_bytes(self):return sum(a.nbytes for a in self.arrays.values())
    @property
    def snapshot_raw_bytes(self):return sum(a[:self.size].nbytes for a in self.arrays.values())+65536

    def offer(self,observation,greedy,**kwargs):
        """Compatibility predicate; offer_index also identifies a replaced slot."""
        return self.offer_index(observation,greedy,**kwargs) is not None

    def offer_index(self,observation,greedy,*,record,policy_version,rng,best_response_mode,
              supplemental=False,enabled=True):
        record=_integer(record,"label record");policy_version=_integer(policy_version,"label policy version")
        if not enabled or supplemental or not best_response_mode or record<self.opening_record:return None
        packed=self.spec.pack(observation);greedy=np.asarray(greedy)
        if greedy.dtype!=np.bool_ or greedy.shape!=(self.spec.actions,) or not greedy.any() or (greedy&~observation.legal).any():
            raise ValueError("frozen greedy targets must be a nonempty subset of legal actions")
        if self.seen>=2**64-1:raise OverflowError("reservoir counter exhausted")
        self.seen+=1
        index=self.size if self.size<self.capacity else int(rng.integers(0,self.seen))
        if index>=self.capacity:return None
        for name,data in zip(("spatial","globals","legal"),packed):self.arrays[name][index]=data
        self.arrays["greedy"][index]=np.packbits(greedy,bitorder="little")
        self.arrays["record"][index]=record;self.arrays["policy_version"][index]=policy_version
        self.size=min(self.size+1,self.capacity);self.revisions+=1
        return index

    def sample(self,count,rng):
        if type(count) is not int or count<1 or not self.size:raise ValueError("positive count and nonempty reservoir required")
        indices=rng.integers(0,self.size,count);a=self.arrays
        spatial,globals,legal=self.spec.unpack(a["spatial"][indices],a["globals"][indices],a["legal"][indices])
        greedy=np.unpackbits(a["greedy"][indices],axis=-1,count=self.spec.actions,bitorder="little").astype(bool)
        targets=greedy.astype(np.float32)/greedy.sum(-1,keepdims=True)
        return {"spatial":spatial,"globals":globals,"legal":legal,"targets":targets.astype(np.float32),
                "records":a["record"][indices].copy(),"policy_version":a["policy_version"][indices].copy(),"indices":indices}

    def save(self,path):
        _save(path,{"format_version":FORMAT_VERSION,"kind":"reservoir","capacity":self.capacity,
                    "spec":asdict(self.spec),"size":self.size,"seen":self.seen,
                    "opening_record":self.opening_record,"revisions":self.revisions},self.arrays,self.size)

    @classmethod
    def load(cls,path,*,max_bytes=DEFAULT_MAX_BYTES):
        meta,arrays=_read(path)
        if meta.get("kind")!="reservoir":raise ValueError("not a reservoir snapshot")
        result=cls(meta["capacity"],ObservationSpec(**meta["spec"]),opening_record=meta["opening_record"],max_bytes=max_bytes)
        seen=_integer(meta["seen"],"seen");size=_integer(meta["size"],"size",maximum=result.capacity)
        if size!=min(seen,result.capacity):raise ValueError("invalid reservoir counts")
        _restore_arrays(result.arrays,arrays,size);result.seen,result.size=seen,size
        result.revisions=_integer(meta.get("revisions",size),"reservoir changes",maximum=seen)
        if result.revisions<size:raise ValueError("invalid reservoir change counter")
        if size:
            if not np.isfinite(result.arrays["globals"][:size]).all():raise ValueError("nonfinite reservoir globals")
            greedy=np.unpackbits(result.arrays["greedy"][:size],axis=-1,count=result.spec.actions,bitorder="little")
            legal=np.unpackbits(result.arrays["legal"][:size],axis=-1,count=result.spec.actions,bitorder="little")
            if not greedy.any(-1).all() or (greedy&~legal.astype(bool)).any():raise ValueError("invalid restored greedy labels")
            if (result.arrays["record"][:size]<result.opening_record).any():raise ValueError("pre-opening labels in reservoir")
        return result
