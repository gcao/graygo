"""Explicit, checksummed optional C++ batch decoding of canonical rule packets."""
import ctypes as ct
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
from .encoding import ENCODING_VERSION,SPATIAL_CHANNELS,GLOBAL_FEATURES

ABI_VERSION=1


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda:file.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


class NativeRuleDecoder:
    def __init__(self,path,expected_sha256=None):
        self.path=str(Path(path).expanduser().resolve());self.sha256=file_sha256(self.path)
        if expected_sha256 is not None and self.sha256!=expected_sha256:raise ValueError('native decoder checksum differs')
        if ENCODING_VERSION!='graygo/encoding/1' or (SPATIAL_CHANNELS,GLOBAL_FEATURES)!=(30,14):raise ValueError('native decoder does not support this observation version')
        try:
            self.library=ct.CDLL(self.path)
            abi=self.library.graygo_rule_decode_abi;abi.argtypes=[];abi.restype=ct.c_uint
            if abi()!=ABI_VERSION:raise ValueError('native rule decoder ABI differs')
            self.function=self.library.graygo_decode_rule_states
        except (OSError,AttributeError) as exc:raise ValueError('invalid native rule decoder library') from exc
        byte=ct.POINTER(ct.c_uint8)
        self.function.argtypes=[byte,ct.c_size_t,ct.c_size_t,byte,ct.c_int,ct.c_uint64,byte,ct.c_size_t,
                               ct.c_int,ct.c_int,byte,ct.POINTER(ct.c_float),byte,ct.POINTER(ct.c_size_t)]
        self.function.restype=ct.c_int

    def decode(self,codec,packets,valid=None):
        rules=codec.rules
        if not 3<=rules.size<=19 or rules.max_rounds>2**32-1:raise ValueError('native decoder supports sizes 3..19 and 32-bit round limits')
        packets=np.asarray(packets)
        if packets.dtype!=np.uint8 or packets.ndim!=2 or packets.shape[1]!=codec.packet_bytes:raise ValueError('native input must be canonical packet rows')
        packets=np.ascontiguousarray(packets)
        if valid is None:valid=np.ones(len(packets),dtype=bool)
        valid=np.asarray(valid)
        if valid.dtype!=np.bool_ or valid.shape!=(len(packets),):raise ValueError('native validity mask must match rows')
        flags=np.ascontiguousarray(valid,dtype=np.uint8);prefix=np.frombuffer(codec.prefix,dtype=np.uint8)
        x=np.empty((len(packets),30,rules.size,rules.size),dtype=np.uint8)
        g=np.empty((len(packets),14),dtype=np.float32);legal=np.empty((len(packets),rules.points+1),dtype=bool)
        if not len(packets):return x,g,legal
        byte=ct.POINTER(ct.c_uint8);bad=ct.c_size_t(0)
        status=self.function(packets.ctypes.data_as(byte),len(packets),codec.packet_bytes,flags.ctypes.data_as(byte),rules.size,rules.max_rounds,
            prefix.ctypes.data_as(byte),len(prefix),codec.round_bytes,codec.action_bytes,
            x.ctypes.data_as(byte),g.ctypes.data_as(ct.POINTER(ct.c_float)),legal.ctypes.data_as(byte),ct.byref(bad))
        if status:raise ValueError(f'native rule decode failed: status={status}, row={bad.value}')
        return x,g,legal


def build_native(output_dir,*,compiler='c++'):
    """Compile owned source into an immutable content-addressed library + manifest."""
    if os.name!='posix':raise ValueError('native builder currently requires a Unix C++17 compiler')
    source=Path(__file__).parent/'native/rule_decode.cpp'
    version=subprocess.run([compiler,'--version'],check=True,capture_output=True,text=True).stdout.strip()
    flags=['-O3','-std=c++17','-fPIC','-fvisibility=hidden','-Wall','-Wextra','-Wpedantic']
    flags+=['-dynamiclib'] if sys.platform=='darwin' else ['-shared']
    identity={'abi':ABI_VERSION,'cpp_sha256':file_sha256(source),'compiler':compiler,'compiler_version':version,
              'flags':flags,'system':platform.system(),'machine':platform.machine(),'encoding_version':ENCODING_VERSION}
    build_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    root=Path(output_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    destination=root/build_id[:20]
    name='libgraygo_rule_decode'+('.dylib' if sys.platform=='darwin' else '.so')
    target=destination/name;manifest=destination/'build.json'
    started=time.monotonic()
    def sync_directory(path):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    with (root/'.build.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if destination.exists():
            if destination.is_symlink():raise ValueError('native build directory cannot be a symlink')
            saved=json.loads(manifest.read_text())
            if saved['build_id']!=build_id or saved['library_sha256']!=file_sha256(target):raise ValueError('existing native build identity differs')
            NativeRuleDecoder(target,saved['library_sha256'])
            return dict(saved,reused=True)
        stage=root/('.build-'+uuid.uuid4().hex);stage.mkdir()
        try:
            command=[compiler,*flags,str(source),'-o',str(stage/name)]
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode:raise RuntimeError('native build failed: '+result.stderr[-8000:])
            checksum=file_sha256(stage/name);NativeRuleDecoder(stage/name,checksum)
            data=dict(identity,build_id=build_id,library=str(target),library_sha256=checksum,build_seconds=time.monotonic()-started,command=command,diagnostics=result.stderr)
            with (stage/'build.json').open('x') as file:json.dump(data,file,sort_keys=True,indent=2);file.flush();os.fsync(file.fileno())
            with (stage/name).open('rb') as file:os.fsync(file.fileno())
            sync_directory(stage)
            # Publish code and metadata together; never replace a loaded build.
            os.rename(stage,destination);sync_directory(root)
            return dict(data,reused=False)
        finally:
            if stage.exists():shutil.rmtree(stage)
