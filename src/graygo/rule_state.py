"""Compact versioned board observations for replay.

Full v2 ko history belongs to active game checkpoints. Hint-enabled packets
store two conditional masks at collection so native replay decoding needs no
history simulation. The original v1 observation/packet layout is preserved.
"""
from dataclasses import asdict,dataclass
from functools import cached_property
import hashlib
import json
import numpy as np
from .engine import BLACK,WHITE,RULE_VERSION,Rules,State,RoundRecord
from .encoding import ENCODING_VERSION,KO_ENCODING_VERSION,SPATIAL_CHANNELS,GLOBAL_FEATURES,encode

CODEC_VERSION='graygo/rule-state/1'


class RuleStateCodec:
    def __init__(self,rules=Rules(),ko_hints=False):
        if not isinstance(rules,Rules):raise ValueError('immutable Rules required')
        if type(ko_hints) is not bool or (ko_hints and rules.ko!="full-history"):raise ValueError("ko hints require explicit v2 rules")
        self.rules=rules;self.points=rules.points;self.ko_hints=ko_hints
        self.board_bytes=(2*self.points+7)//8;self.mask_bytes=(self.points+7)//8
        self.action_bytes=max(1,(self.points.bit_length()+7)//8)
        self.round_bytes=max(1,(rules.max_rounds.bit_length()+7)//8)
        fingerprint=hashlib.sha256(json.dumps({'codec':CODEC_VERSION,'rule_version':rules.rule_version,'rules':rules.to_dict()},sort_keys=True,separators=(',',':')).encode()).digest()[:8]
        if ko_hints:fingerprint=hashlib.sha256(fingerprint+b'ko-hints/2').digest()[:8]
        self.prefix=(b'GG\x02' if ko_hints else b'GG\x01')+fingerprint
        self.header_bytes=len(self.prefix)+3+self.round_bytes
        self.base_packet_bytes=self.header_bytes+3*self.board_bytes+2*self.mask_bytes+4*self.action_bytes
        self.packet_bytes=self.base_packet_bytes+(2*self.mask_bytes if ko_hints else 0)

    def _colors(self,board):
        a=np.asarray(board,dtype=np.uint8);packed=np.zeros(self.board_bytes,dtype=np.uint8)
        for offset in range(4):packed[:len(a[offset::4])]|=a[offset::4]<<(2*offset)
        return packed.tobytes()

    def _board(self,data):
        if self.points%4 and data[-1]>>(2*(self.points%4)):raise ValueError('noncanonical board padding')
        a=np.frombuffer(data,dtype=np.uint8)
        return tuple(map(int,((a[:,None]>>np.array([0,2,4,6],dtype=np.uint8))&3).reshape(-1)[:self.points]))

    def _forbidden(self,points):
        mask=np.zeros(self.points,dtype=bool);mask[list(points)]=True
        return np.packbits(mask,bitorder='little').tobytes()

    def _points(self,data):
        if self.points%8 and data[-1]>>(self.points%8):raise ValueError('noncanonical restriction padding')
        return frozenset(map(int,np.flatnonzero(np.unpackbits(np.frombuffer(data,dtype=np.uint8),count=self.points,bitorder='little'))))

    def pack(self,state,player):
        if not isinstance(state,State) or state.rules!=self.rules:raise ValueError('state rules disagree with packet codec')
        if type(player) is not int or player not in (BLACK,WHITE):raise ValueError('black/white perspective required')
        header=self.prefix+bytes((player,len(state.history),state.pass_streak))+state.rounds_played.to_bytes(self.round_bytes,'little')
        boards=[bytes(self.board_bytes)]*2;moves=[bytes(2*self.action_bytes)]*2
        for slot,h in enumerate(state.history,start=2-len(state.history)):
            boards[slot]=self._colors(h.before)
            moves[slot]=h.black.to_bytes(self.action_bytes,'little')+h.white.to_bytes(self.action_bytes,'little')
        packet=b''.join((header,self._colors(state.board),*boards,self._forbidden(state.forbidden_black),self._forbidden(state.forbidden_white),*moves))
        if self.ko_hints:
            from .engine import solo_repeat_hints
            packet+=b''.join(self._forbidden(points) for points in solo_repeat_hints(state))
        return packet

    def unpack(self,payload):
        if not isinstance(payload,(bytes,bytearray,memoryview)):
            array=np.asarray(payload)
            if array.dtype!=np.uint8 or array.shape!=(self.packet_bytes,):raise ValueError('packet must be a byte row')
            payload=array.tobytes()
        else:payload=bytes(payload)
        if len(payload)!=self.packet_bytes or not payload.startswith(self.prefix):raise ValueError('packet length, version or rules fingerprint differs')
        cursor=len(self.prefix);player,count,passes=payload[cursor:cursor+3];cursor+=3
        if player not in (BLACK,WHITE) or count>2 or passes>2:raise ValueError('invalid packet perspective/history/pass count')
        rounds=int.from_bytes(payload[cursor:cursor+self.round_bytes],'little');cursor+=self.round_bytes
        boards=[];raw_boards=[]
        for _ in range(3):
            raw=payload[cursor:cursor+self.board_bytes];cursor+=self.board_bytes;raw_boards.append(raw);boards.append(self._board(raw))
        restrictions=[]
        for _ in range(2):restrictions.append(self._points(payload[cursor:cursor+self.mask_bytes]));cursor+=self.mask_bytes
        history=[]
        for slot in range(2):
            raw=payload[cursor:cursor+2*self.action_bytes];cursor+=2*self.action_bytes
            if slot<2-count:
                if any(raw) or any(raw_boards[slot+1]):raise ValueError('noncanonical absent history slot')
            else:history.append(RoundRecord(boards[slot+1],int.from_bytes(raw[:self.action_bytes],'little'),int.from_bytes(raw[self.action_bytes:],'little')))
        state=State(self.rules,boards[0],*restrictions,tuple(history),passes,rounds,
                    None if self.rules.ko=="full-history" else ())
        if self.ko_hints:self.hints(payload) # Validate canonical padding on decode.
        return state,player

    def hints(self,payload):
        if not self.ko_hints:raise ValueError("packet has no ko hints")
        raw=bytes(payload);start=self.base_packet_bytes
        return tuple(self._points(raw[start+i*self.mask_bytes:start+(i+1)*self.mask_bytes]) for i in range(2))


@dataclass(frozen=True)
class RuleObservation:
    """Features bound to one immutable public state and observation version."""
    state: State
    player: int
    ko_hints: bool=False

    def __post_init__(self):
        if not isinstance(self.state,State) or type(self.player) is not int or self.player not in (BLACK,WHITE):raise ValueError('public State and black/white perspective required')

    @cached_property
    def _features(self):return encode(self.state,self.player,ko_hints=self.ko_hints)
    @property
    def spatial(self):return self._features.spatial
    @property
    def global_features(self):return self._features.global_features
    @cached_property
    def legal(self):
        mask=np.zeros(self.state.rules.points+1,dtype=bool)
        mask[list(self.state.legal_actions(self.player))]=True
        mask.flags.writeable=False
        return mask


@dataclass(frozen=True)
class RuleStateSpec:
    rules: Rules=Rules()
    native_library: str|None=None
    native_sha256: str|None=None
    ko_hints: bool=False

    def __post_init__(self):
        if type(self.ko_hints) is not bool or (self.ko_hints and self.rules.ko!='full-history'):raise ValueError('ko hints require v2 rules')
        if not isinstance(self.rules,Rules) or self.rules.points+1>65536:raise ValueError('rules exceed compact-buffer action range')
        if self.native_library is None:
            if self.native_sha256 is not None:raise ValueError('native checksum requires a library')
        else:
            from pathlib import Path
            from .native_decode import file_sha256
            path=str(Path(self.native_library).expanduser().resolve());checksum=file_sha256(path)
            if self.native_sha256 is not None and checksum!=self.native_sha256:raise ValueError('native decoder checksum differs')
            if not 3<=self.rules.size<=19 or self.rules.max_rounds>2**32-1:raise ValueError('native decoder does not support these rule dimensions')
            object.__setattr__(self,'native_library',path);object.__setattr__(self,'native_sha256',checksum)

    @cached_property
    def codec(self):return RuleStateCodec(self.rules,self.ko_hints)
    @property
    def spatial_shape(self):return (32 if self.ko_hints else SPATIAL_CHANNELS,self.rules.size,self.rules.size)
    @property
    def globals(self):return GLOBAL_FEATURES
    @property
    def actions(self):return self.rules.points+1
    @property
    def version(self):return KO_ENCODING_VERSION if self.ko_hints else ENCODING_VERSION
    @property
    def spatial_bytes(self):return self.codec.packet_bytes
    @property
    def legal_bytes(self):return (self.actions+7)//8
    @property
    def stored_globals(self):return 0
    @property
    def stored_legal_bytes(self):return 0

    @cached_property
    def native_decoder(self):
        if self.native_library is None:return None
        from .native_decode import NativeRuleDecoder
        return NativeRuleDecoder(self.native_library,self.native_sha256)

    def to_dict(self):
        value={'storage':'rule-state','codec':CODEC_VERSION,'rule_version':self.rules.rule_version,'encoding_version':ENCODING_VERSION,'rules':self.rules.to_dict()}
        if self.ko_hints:value.update(codec='graygo/rule-state/2',encoding_version=KO_ENCODING_VERSION,ko_hints=True)
        if self.native_library is not None:value['native_decoder']={'path':self.native_library,'sha256':self.native_sha256,'abi':1}
        return value
    @classmethod
    def from_dict(cls,value):
        if value.get('ko_hints') is True:
            raw=dict(value);raw.pop('ko_hints');raw['codec']=CODEC_VERSION;raw['encoding_version']=ENCODING_VERSION
            if value.get('codec')!='graygo/rule-state/2' or value.get('encoding_version')!=KO_ENCODING_VERSION:raise ValueError('ko observation schema differs')
            base=cls.from_dict(raw);return cls(base.rules,base.native_library,base.native_sha256,True)
        fields={'storage','codec','rule_version','encoding_version','rules'}
        if set(value) not in (fields,fields|{'native_decoder'}) or value['storage']!='rule-state' or value['codec']!=CODEC_VERSION or value['rule_version']!=Rules(**value['rules']).rule_version or value['encoding_version']!=ENCODING_VERSION:
            raise ValueError('unsupported rule-state storage schema')
        if 'native_decoder' not in value:return cls(Rules(**value['rules']))
        native=value['native_decoder']
        if not isinstance(native,dict) or set(native)!={'path','sha256','abi'} or type(native['abi']) is not int or native['abi']!=1 or not isinstance(native['path'],str) or not native['path'] or not isinstance(native['sha256'],str) or len(native['sha256'])!=64 or any(c not in '0123456789abcdef' for c in native['sha256']):raise ValueError('unsupported native decoder descriptor')
        return cls(Rules(**value['rules']),native['path'],native['sha256'])

    def pack(self,observation):
        if not isinstance(observation,RuleObservation):raise ValueError('rule-state storage requires a state-bound RuleObservation')
        if observation.ko_hints!=self.ko_hints:raise ValueError("observation and replay hint versions differ")
        payload=self.codec.pack(observation.state,observation.player)
        return np.frombuffer(payload,dtype=np.uint8),np.empty(0,dtype=np.float32),np.empty(0,dtype=np.uint8)

    def _rows(self,spatial,globals,legal,valid=None):
        leading=spatial.shape[:-1]
        if spatial.dtype!=np.uint8 or spatial.shape[-1:]!=(self.spatial_bytes,) or globals.shape!=(*leading,0) or legal.shape!=(*leading,0):raise ValueError('stored rule-state dimensions disagree')
        if valid is None:valid=np.ones(leading,dtype=bool)
        if valid.dtype!=np.bool_ or valid.shape!=leading:raise ValueError('valid rows must match packet batch')
        return leading,spatial.reshape(-1,self.spatial_bytes),valid.reshape(-1)

    def unpack(self,spatial,globals,legal,*,valid=None):
        leading,rows,valid=self._rows(spatial,globals,legal,valid)
        if self.native_decoder is not None:
            x,g,masks=self.native_decoder.decode(self.codec,rows,valid)
            return x.reshape(*leading,*self.spatial_shape),g.reshape(*leading,self.globals),masks.reshape(*leading,self.actions)
        x=np.zeros((*leading,*self.spatial_shape),dtype=np.uint8);g=np.zeros((*leading,self.globals),dtype=np.float32);masks=np.zeros((*leading,self.actions),dtype=bool)
        flat_x=x.reshape(-1,*self.spatial_shape);flat_g=g.reshape(-1,self.globals);flat_masks=masks.reshape(-1,self.actions)
        for index,row in enumerate(rows):
            if not valid[index]:continue
            state,player=self.codec.unpack(row);features=encode(state,player,ko_hints=self.ko_hints,hint_points=self.codec.hints(row) if self.ko_hints else None)
            flat_x[index]=features.spatial;flat_g[index]=features.global_features;flat_masks[index]=features.legal
        return x,g,masks

    def legal_masks(self,spatial,globals,legal):
        leading,rows,valid=self._rows(spatial,globals,legal)
        masks=np.zeros((*leading,self.actions),dtype=bool);flat=masks.reshape(-1,self.actions)
        if self.native_decoder is not None:
            for start in range(0,len(rows),1024):
                _,_,mask=self.native_decoder.decode(self.codec,rows[start:start+1024])
                flat[start:start+len(mask)]=mask
        else:
            for index,row in enumerate(rows):
                state,player=self.codec.unpack(row);flat[index,list(state.legal_actions(player))]=True
        return masks
