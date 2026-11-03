"""Checked specialization of the optional actor pool for unchanged A2 callers.

Parses source as data. A proof covers valid default-pool snapshot states and
the pinned task callers, not arbitrary external callers, mutations or graphs.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tools.task_source_diff import inventory, dependency_graph, normalized

VERSION = 'graygo/default-task-actor-specialization/1'
DELEGATE = '''def forward(self,kind,**inputs):
    if kind not in ('q','average'):raise ValueError('unknown frozen forward policy')
    model=self.q if kind=='q' else self.average
    if self.forward_pool is None:return model(**inputs)
    return self.forward_pool.forward(model,inputs['spatial'],inputs['globals'])
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    return normalized(left) == normalized(right)


def expression(text):
    return ast.parse(text, mode='eval').body


def ast_sha(node):
    return hashlib.sha256(normalized(node).encode()).hexdigest()


def default_callers(files):
    graph = dependency_graph(files, ('nfsp_training', 'task_training'))
    calls = []
    for module in graph['modules']:
        if module in ('nfsp_collection', 'frozen_forward'):
            continue
        tree = ast.parse(files[module + '.py'])
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in ('nfsp_collection', 'graygo.nfsp_collection'):
                for alias in node.names:
                    require(alias.name != 'FrozenPolicies' or alias.asname is None, 'aliased snapshot constructor requires review')
            if isinstance(node, ast.Name) and node.id == 'FrozenPolicies' and isinstance(node.ctx, ast.Store):
                raise ValueError('snapshot constructor binding is reassigned')
            if isinstance(node, ast.Attribute) and node.attr == 'FrozenPolicies':
                raise ValueError('qualified snapshot constructor escapes its checked call sites')
            if isinstance(node, ast.Constant) and node.value == 'forward_pool':
                raise ValueError('A2 consumer can access optional pool state dynamically')
            if isinstance(node, ast.Name) and node.id == 'FrozenPolicies' and isinstance(node.ctx, ast.Load):
                parent = parents.get(node)
                require(isinstance(parent, ast.Call) and parent.func is node, 'snapshot constructor escapes its checked call sites')
                require(len(parent.args) == 2 and all(k.arg in ('version', 'average_ready') for k in parent.keywords)
                        and {k.arg for k in parent.keywords} == {'version', 'average_ready'}, 'A2 caller can select a pool or hide constructor arguments')
                calls.append({'module': module, 'line': node.lineno, 'call': normalized(parent)})
            if isinstance(node, ast.Attribute) and node.attr == 'forward_pool':
                raise ValueError('A2 consumer observes or mutates the optional pool')
    require(len(calls) == 2 and all(row['module'] == 'task_training' for row in calls), 'expected exactly the live and restored default task snapshots')
    return sorted(calls, key=lambda row: (row['module'], row['line']))


class Specialize(ast.NodeTransformer):
    def __init__(self):
        self.counts = {'inlined_forwards': 0, 'pool_assignment': 0, 'pool_argument': 0,
                       'false_branches': 0, 'metadata_get': 0, 'identity_false': 0,
                       'identity_import': 0, 'state_return': 0, 'delegate_removed': 0}
        self.method = None

    def visit_ClassDef(self, node):
        if node.name != 'FrozenPolicies':
            return self.generic_visit(node)
        delegate = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'forward'), None)
        require(delegate is not None and same(delegate, ast.parse(DELEGATE).body[0]), 'optional forward delegate differs from the checked direct-call identity')
        node.body.remove(delegate); self.counts['delegate_removed'] += 1
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            self.method = method.name
            if method.name == '__init__':
                args = method.args
                indices = [i for i, arg in enumerate(args.kwonlyargs) if arg.arg == 'forward_pool']
                require(len(indices) == 1 and same(args.kw_defaults[indices[0]], expression('None')), 'default pool must be literal None')
                index = indices[0]; args.kwonlyargs.pop(index); args.kw_defaults.pop(index)
                self.counts['pool_argument'] += 1
            self.generic_visit(method)
            if method.name == 'state_dict':
                require(len(method.body) == 2 and isinstance(method.body[0], ast.Assign)
                        and len(method.body[0].targets) == 1 and isinstance(method.body[0].targets[0], ast.Name)
                        and method.body[0].targets[0].id == 'result'
                        and isinstance(method.body[0].value, ast.Dict) and isinstance(method.body[1], ast.Return)
                        and same(method.body[1].value, expression('result')), 'unexpected default snapshot serialization')
                method.body = [ast.Return(value=method.body[0].value)]
                self.counts['state_return'] += 1
        self.method = None
        return node

    def visit_Assign(self, node):
        target = node.targets[0] if len(node.targets) == 1 else None
        if self.method == '__init__' and isinstance(target, ast.Attribute) and target.attr == 'forward_pool' and same(target.value, expression('self')):
            require(same(node.value, expression('forward_pool')), 'pool field has another assignment')
            self.counts['pool_assignment'] += 1
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        if self.method and node.id == 'forward_pool' and isinstance(node.ctx, ast.Load):
            return ast.Constant(None)
        return node

    def visit_Attribute(self, node):
        if self.method and same(node, expression('self.forward_pool')):
            return ast.Constant(None)
        return self.generic_visit(node)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.left, ast.Constant) and isinstance(node.comparators[0], ast.Constant):
            left, right = node.left.value, node.comparators[0].value
            require(left is None or type(left) is bool, 'unexpected scalar specialization')
            require(right is None or type(right) is bool, 'unexpected scalar specialization')
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.Eq)):
                return ast.Constant(left == right)
            if isinstance(op, (ast.IsNot, ast.NotEq)):
                return ast.Constant(left != right)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.And) and isinstance(node.values[0], ast.Constant) and node.values[0].value is False:
            return ast.Constant(False)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            self.counts['false_branches'] += 1
            return node.orelse
        return node

    def visit_Call(self, node):
        # All routed inference calls have a literal valid kind and unchanged kwargs.
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'forward' and isinstance(node.func.value, (ast.Name, ast.Attribute)):
            receiver = node.func.value
            if same(receiver, expression('snapshot')) or same(receiver, expression('self.snapshot')):
                require(len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value in ('q', 'average')
                        and len(node.keywords) == 1 and node.keywords[0].arg is None and same(node.keywords[0].value, expression('inputs')), 'routed forward has different arguments')
                self.counts['inlined_forwards'] += 1
                return ast.Call(func=ast.Attribute(value=receiver, attr=node.args[0].value, ctx=ast.Load()), args=[], keywords=node.keywords)
        node = self.generic_visit(node)
        if self.method == 'load_state_dict' and same(node, expression("state.get('frozen_execution')")):
            self.counts['metadata_get'] += 1
            return ast.Constant(None)
        if self.method == 'load_state_dict' and same(node, expression('execution_identity(False)')):
            self.counts['identity_false'] += 1
            return ast.Constant(None)
        return node

    def visit_ImportFrom(self, node):
        if self.method == 'load_state_dict' and same(node, ast.parse('from .frozen_forward import execution_identity').body[0]):
            self.counts['identity_import'] += 1
            return None
        return node


def specialize(source):
    tree = ast.parse(source); transformer = Specialize(); specialized = transformer.visit(tree)
    expected = {'inlined_forwards': 4, 'pool_assignment': 1, 'pool_argument': 1, 'false_branches': 4,
                'metadata_get': 1, 'identity_false': 1, 'identity_import': 1, 'state_return': 1, 'delegate_removed': 1}
    require(transformer.counts == expected, 'specialization rule inventory differs: ' + str(transformer.counts))
    return specialized, transformer.counts


def prove(reference, candidate, *, expected_reference, expected_candidate):
    old_root, old, old_sha = inventory(reference); new_root, new, new_sha = inventory(candidate)
    require(old_sha == expected_reference and new_sha == expected_candidate, 'pinned package differs')
    require(old['task_training.py'] == new['task_training.py'] and old['nfsp_training.py'] == new['nfsp_training.py'], 'task callers or matrix trainer changed')
    callers_old, callers_new = default_callers(old), default_callers(new)
    require(callers_old == callers_new, 'snapshot call sites differ')
    # Only this function is evaluated abstractly: disabled execution is None.
    identity = next(n for n in ast.parse(new['frozen_forward.py']).body if isinstance(n, ast.FunctionDef) and n.name == 'execution_identity')
    require(len(identity.body) == 3 and same(identity.body[0], ast.parse("if type(enabled) is not bool:raise ValueError('explicit frozen CUDA graph selection required')").body[0])
            and same(identity.body[1], ast.parse('if not enabled:return None').body[0]) and isinstance(identity.body[2], ast.Return), 'disabled execution identity has another observable path')
    transformed, rules = specialize(new['nfsp_collection.py'])
    original = ast.parse(old['nfsp_collection.py'])
    require(same(original, transformed), 'default collector still differs after the checked specialization')
    require(inventory(old_root)[1:] == (old, old_sha) and inventory(new_root)[1:] == (new, new_sha), 'source changed during proof')
    return {'version': VERSION, 'status': 'default_collection_ast_equivalent', 'reference_source_sha256': old_sha,
            'candidate_source_sha256': new_sha, 'default_constructor_sites': callers_new, 'specialization_rules': rules,
            'reference_collection_ast_sha256': ast_sha(original), 'specialized_candidate_ast_sha256': ast_sha(transformed),
            'conditions': ['Pinned unmodified task callers construct both live/restored snapshots without an optional pool.',
                           'Valid default/old snapshot schema contains q, average, version and average_ready; frozen_execution is absent.',
                           'No external mutation, monkey-patching, graph snapshot or caller outside the pinned task dependency closure.',
                           'Disabled execution-identity import has separately verified unchanged numerical RNG behavior.'],
            'scope': 'Direct default collector forward/serialization/restore specialization only; game-domain, runtime, numerical and full-cohort evidence remain separate.',
            'full_source_compatibility_qualified': False, 'a2_evidence_reusable': False, 'execution_authorized': False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('reference', 'candidate', 'expected-reference', 'expected-candidate', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        out = Path(args.output).resolve()
        require(not out.exists() and all(not out.is_relative_to(Path(root).resolve()) for root in (args.reference, args.candidate)), 'new output outside input packages required')
        result = prove(args.reference, args.candidate, expected_reference=args.expected_reference, expected_candidate=args.expected_candidate)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
        print(json.dumps(result, indent=2)); return 0
    except (ValueError, OSError, SyntaxError, StopIteration) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
