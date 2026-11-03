"""Recover the literal pre-observation target module after checked additions."""
import ast
from copy import deepcopy
from tools.nfsp_resources import require


def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def prove(reference,candidate):
    require(set(reference)==set(candidate),'application file inventory changed')
    changed={name for name in reference if reference[name]!=candidate[name]}
    require(changed=={'sequence_targets.py'},'unreviewed application file changed')
    before=ast.parse(reference['sequence_targets.py']);after=ast.parse(candidate['sequence_targets.py'])
    tree=deepcopy(after)
    functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    targets=functions['retrace_targets']
    assignments=[node for node in targets.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='details' for t in node.targets)]
    require(len(assignments)==1 and isinstance(assignments[0].value,ast.Dict),'one detached detail dictionary required')
    values=assignments[0].value
    indices=[i for i,key in enumerate(values.keys) if isinstance(key,ast.Constant) and key.value=='raw_targets']
    require(len(indices)==1,'one new raw-target detail required')
    expected=ast.parse('raw.cpu().double().numpy()',mode='eval').body
    index=indices[0];require(same(values.values[index],expected),'raw-target observation is not a detached copy')
    del values.keys[index];del values.values[index]
    update=functions['update_retrace_q']
    args=update.args
    positions=[i for i,arg in enumerate(args.kwonlyargs) if arg.arg=='target_observer']
    require(len(positions)==1 and isinstance(args.kw_defaults[positions[0]],ast.Constant)
            and args.kw_defaults[positions[0]].value is None,'optional observer must default to None')
    index=positions[0];del args.kwonlyargs[index];del args.kw_defaults[index]
    expected=ast.parse("if target_observer is not None:"+chr(10)+"    target_observer(labels.detach(), stats, details)").body[0]
    nodes=[i for i,node in enumerate(update.body) if same(node,expected)]
    require(len(nodes)==1,'exact nonmutating target observation call required')
    del update.body[nodes[0]]
    require(same(before,tree),'original target/gradient code was not recovered after only declared observation additions')
    return {'default_target_and_update_AST_recovered':True,'all_other_application_bytes_identical':True,
            'source_admission_qualified':False,'runtime_prices_reused':False,
            'scope':'literal checked extension for detached observation only; callback behavior has separate actual pipeline parity evidence'}
