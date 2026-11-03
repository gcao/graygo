"""Recover the complete literal pre-cache module after checked cache additions."""
import ast
from copy import deepcopy
from tools.terminal_batch_admission import prove_storage_and_collector
from tools.nfsp_resources import require


def tree(value):return ast.parse(value)
def same(left,right):return ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
def statement(source):return tree(source).body[0]


def prove(reference,candidate):
    names=set(reference)|set(candidate);changed={n for n in names if reference.get(n)!=candidate.get(n)}
    require(changed=={'buffers.py','nfsp_training.py','nfsp_collection.py','cpu_frozen_forward.py'},'unreviewed application module changed')
    storage=prove_storage_and_collector({n:reference[n] for n in ('buffers.py','nfsp_training.py')},{n:candidate[n] for n in ('buffers.py','nfsp_training.py')})
    expected=tree(reference['nfsp_collection.py']);actual=tree(candidate['nfsp_collection.py']);wanted=statement('from .cpu_frozen_forward import CPU3x3ForwardCache,supports_model')
    imports=[n for n in actual.body if isinstance(n,ast.ImportFrom) and n.module=='cpu_frozen_forward'];require(len(imports)==1 and same(imports[0],wanted),'CPU cache imports changed');actual.body.remove(imports[0])
    owner=next(n for n in actual.body if isinstance(n,ast.ClassDef) and n.name=='FrozenPolicies');methods={n.name:n for n in owner.body if isinstance(n,ast.FunctionDef)}
    ctor=methods['__init__'];assign=statement("self._cpu_forward_cache=CPU3x3ForwardCache() if forward_pool is None and self.device.type=='cpu' and supports_model(self.q) and supports_model(self.average) else None")
    additions=[n for n in ctor.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_cpu_forward_cache' for t in n.targets)];require(len(additions)==1 and same(additions[0],assign),'cache constructor changes execution selection');ctor.body.remove(additions[0])
    expanded=statement("if self.forward_pool is None:\n if self._cpu_forward_cache is not None:return self._cpu_forward_cache.forward(model,**inputs)\n return model(**inputs)")
    forward=methods['forward'];branches=[n for n in forward.body if isinstance(n,ast.If) and same(n.test,expanded.test)];require(len(branches)==1 and same(branches[0],expanded),'cache altered model calls or arguments');forward.body[forward.body.index(branches[0])]=statement('if self.forward_pool is None:return model(**inputs)')
    restore=methods['load_state_dict'];invalidation=statement('if self._cpu_forward_cache is not None:self._cpu_forward_cache.invalidate()');matches=[n for n in restore.body if same(n,invalidation)];require(len(matches)==1,'explicit restored-cache invalidation missing');restore.body.remove(matches[0])
    require(same(expected,actual),'original collection/actor/serialization/RNG code changed')
    cache=tree(candidate['cpu_frozen_forward.py']);imports=[n for n in cache.body if isinstance(n,(ast.Import,ast.ImportFrom))];allowed=[statement('from collections import OrderedDict'),statement('from weakref import WeakKeyDictionary'),statement('import torch'),statement('from .models import QNet,PolicyNet')]
    require(len(imports)==len(allowed) and all(same(a,b) for a,b in zip(imports,allowed)),'cache imports an unreviewed computation or RNG source')
    supported=next(n for n in cache.body if isinstance(n,ast.FunctionDef) and n.name=='supports_model');require(same(supported,statement('def supports_model(model):return type(model) in (QNet,PolicyNet)')),'cache expanded supported model classes')
    return {'literal_original_snapshot_actor_collection_and_serialization_ast_recovered':True,'scalar_storage_and_format_preserved':storage['scalar_storage_and_format_preserved'],'matrix_other_semantics_preserved':storage['matrix_other_semantics_preserved'],'new_cache_module_requires_receipt_bound_tests_and_actual_pipeline_equality':True,'checkpoint_schema_changed':False,'source_admission_granted':False,'execution_authorized':False}
