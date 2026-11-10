"""Frozen v2 policy descriptors, sealed archives and arena payloads."""
import io,json,shutil
from pathlib import Path
import torch
from graygo.engine import Rules
from graygo.evaluation import FrozenNeuralPolicy
from graygo.manifest import write_report,source_digest
from graygo.storage import CheckpointStore,digest
from graygo.confidence import canonical_sha
from tools import policy_selection

RULES=Rules(ko='full-history')


def describe(archive,label):
    frozen=FrozenNeuralPolicy.from_archive(archive,label=label,evaluation_rules=RULES)
    selection=policy_selection.selection_spec(0.,True)
    selected=policy_selection.apply_selection(frozen,selection)
    return {'kind':'archive','archive':str(Path(archive).resolve()),'label':label,'records':frozen.identity['origin']['records'],
            'head':'average','rules':RULES.to_dict(),'evaluation_rules':RULES.to_dict(),'selection':selection,
            'weights_sha256':frozen.identity['weights_sha256'],'policy_sha256':selected.sha256}


def export(descriptor,root):
    descriptor=dict(descriptor);source=Path(descriptor['archive']);CheckpointStore.verify(source)
    data=torch.load(source/'models.pt',weights_only=True,map_location='cpu');cfg=data['config']
    folder=Path(root)/descriptor['policy_sha256'];folder.mkdir(parents=True,exist_ok=True)
    target=folder/'policy.pt';hints=cfg.get('history_features',False)
    blob={'format_version':'graygo/arena-nfsp-average/1','head':'average','encoding_version':'graygo/encoding/ko-hints/2' if hints else 'graygo/encoding/1',
          'board_size':cfg['rules']['size'],'rules':RULES.to_dict(),'training_records':descriptor['records'],
          'history_features':hints,'training_rules':cfg['rules'],'model_config':{key:cfg[key] for key in ('width','blocks','norm_groups')},'state_dict':data['average'],
          'selection':descriptor['selection'],'strength_validated':False,'source_archive_sha256':digest(source/'manifest.json')}
    if not target.exists():
        stream=io.BytesIO();torch.save(blob,stream);temporary=folder/'.policy.tmp';temporary.write_bytes(stream.getvalue());temporary.replace(target)
    else:
        stored=torch.load(target,weights_only=True,map_location='cpu')
        if stored['source_archive_sha256']!=blob['source_archive_sha256']:raise ValueError('immutable v2 release origin changed')
    sidecar=folder/'policy_selection.py'
    if not sidecar.exists():shutil.copyfile(policy_selection.__file__,sidecar)
    if digest(sidecar)!=descriptor['selection']['implementation_sha256']:raise ValueError('frozen selector changed')
    return dict(descriptor,bundle=str(target.resolve()),bundle_sha256=digest(target))


def archive_runtime(runtime,root,reserve):
    root=Path(root);store=CheckpointStore(root,reserve_bytes=reserve);path=root/'weights'/f'{runtime.records:012d}'
    if path.exists():
        CheckpointStore.verify(path);saved=torch.load(path/'models.pt',weights_only=True,map_location='cpu')
        for name in ('q','average'):
            if any(not torch.equal(value.cpu(),saved[name][key]) for key,value in getattr(runtime.backend,name).state_dict().items()):raise ValueError('same-record immutable archive differs')
        return path
    stream=io.BytesIO();torch.save({'config':runtime.config.plan(),'q':runtime.backend.q.state_dict(),'average':runtime.backend.average.state_dict()},stream)
    raw=stream.getvalue()
    return store.save(runtime.records,lambda p:(p/'models.pt').write_bytes(raw),kind='weights',
        metadata={'source_sha256':source_digest(),'native_schema':runtime.spec.to_dict(),'initial_weights':runtime.initial_weights},estimated_peak_bytes=len(raw)+65536)


def publish(path,descriptor,state):
    body={'version':'graygo/champion-release/1','champion':descriptor,'experimental':True,'formal_training_or_promotion_grant':False,
          'promotion_policy':'paired-betting/v4','rule_version':'graygo/2','campaign':state['campaign'],'promotions':state['promotions'],
          'gate_attempt':state['gate_attempts']}
    write_report(path,{'body':body,'sha256':canonical_sha(body)})
