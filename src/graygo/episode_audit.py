"""Append-only fixed-schema episode evidence with checkpointed prefix integrity.

An uncommitted tail is discarded only on explicit trainer resume. The immutable
checkpoint stores the committed byte count and SHA-256; a corrupt prefix fails.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

EPISODE_DTYPE=np.dtype([
    ('episode','<u8'),('record','<u8'),('generation','<u8'),('average_ready','?'),
    ('best_response','u1',(2,)),('acting','u1',(2,)),('mode_draw','<f8',(2,)),
    ('ramp_draw','<f8',(2,)),('ramp_probability','<f8',(2,)),('epsilon','<f8',(2,)),
    ('actions','<u2',(2,)),('behavior_prob','<f8',(2,)),('reward_black','<f8')])


class EpisodeAudit:
    def __init__(self,root,*,create=False,dtype=EPISODE_DTYPE):
        self.dtype=np.dtype(dtype)
        self.path=Path(root)/'episodes.bin';schema=Path(root)/'episodes.schema.json'
        expected={'format_version':1,'dtype':json.loads(json.dumps(self.dtype.descr)),
                  'itemsize':self.dtype.itemsize,'acting_codes':{'q':0,'average':1,'uniform':2}}
        if create:
            with self.path.open('xb'):pass
            with schema.open('x') as file:json.dump(expected,file,sort_keys=True,indent=2)
        elif json.loads(schema.read_text())!=expected:raise ValueError('episode audit schema differs')

    def append(self,rows):
        if rows.dtype!=self.dtype or rows.ndim!=1:raise ValueError('invalid episode audit rows')
        with self.path.open('ab') as file:file.write(rows.tobytes())

    def prefix(self,episodes):
        size=episodes*self.dtype.itemsize
        if self.path.stat().st_size!=size:raise ValueError('episode audit length differs from physical episodes')
        with self.path.open('rb') as file:os.fsync(file.fileno())
        return {'episodes':episodes,'bytes':size,'sha256':self._digest(size)}

    def _digest(self,size):
        h=hashlib.sha256();remaining=size
        with self.path.open('rb') as file:
            while remaining:
                data=file.read(min(1024*1024,remaining))
                if not data:raise ValueError('episode audit prefix is truncated')
                h.update(data);remaining-=len(data)
        return h.hexdigest()

    def restore_prefix(self,state):
        if type(state['episodes']) is not int or state['episodes']<0 or state['bytes']!=state['episodes']*self.dtype.itemsize:
            raise ValueError('invalid episode audit checkpoint counters')
        size=state['bytes']
        if self.path.stat().st_size<size or self._digest(size)!=state['sha256']:raise ValueError('committed episode audit prefix is corrupt')
        with self.path.open('r+b') as file:file.truncate(size);file.flush();os.fsync(file.fileno())
