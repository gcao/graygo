"""Read-only dependency/source inventory; never qualifies old A2 evidence.

No compared package is imported. Conditional imports are followed conservatively;
dynamic import/execution sites and external numerical dependencies stay visible.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

VERSION = 'graygo/task-source-diff/1'
ENTRIES = ('nfsp_training', 'task_training')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(root):
    root = Path(root).resolve()
    require(root.is_dir(), 'package directory required')
    paths = sorted([*root.glob('*.py'), *(root / 'native').glob('*.cpp')])
    require((root / '__init__.py') in paths, 'flat graygo package required')
    require(not any(path.is_symlink() for path in paths), 'source files cannot be symlinks')
    files = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    h = hashlib.sha256()
    for name, content in files.items():
        h.update(name.encode() + b'\0' + content + b'\0')
    return root, files, h.hexdigest()


def tree(content):
    return ast.parse(content)


def normalized(node):
    return ast.dump(node, include_attributes=False)


def definitions(module):
    result = {}
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = normalized(node)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result[node.name + '.' + method.name] = normalized(method)
    return result


def dependency_graph(files, entries):
    todo = ['__init__', *entries]
    modules, external, dynamic = {}, set(), []
    while todo:
        name = todo.pop()
        if name in modules:
            continue
        require(name + '.py' in files, 'missing local dependency: ' + name)
        module = tree(files[name + '.py'])
        children = set()
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    require(node.level == 1, 'flat-package relative import required')
                    if node.module:
                        children.add(node.module)
                    else:
                        children.update(alias.name for alias in node.names if alias.name + '.py' in files)
                elif node.module == 'graygo':
                    children.add('__init__')
                    children.update(alias.name for alias in node.names if alias.name + '.py' in files)
                elif node.module and node.module.startswith('graygo.'):
                    children.add(node.module[7:])
                elif node.module:
                    external.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'graygo':
                        children.add('__init__')
                    elif alias.name.startswith('graygo.'):
                        children.add(alias.name[7:])
                    else:
                        external.add(alias.name)
            elif isinstance(node, ast.Call):
                call = node.func
                terminal = call.id if isinstance(call, ast.Name) else call.attr if isinstance(call, ast.Attribute) else None
                if terminal in ('__import__', 'import_module', 'eval', 'exec', 'run_path', 'run_module'):
                    dynamic.append({'module': name, 'line': node.lineno, 'call': normalized(node)})
                    # Literal local dynamic imports still belong to the closure.
                    if terminal in ('__import__', 'import_module') and node.args and isinstance(node.args[0], ast.Constant):
                        target = node.args[0].value
                        if isinstance(target, str) and target.startswith('graygo.'):
                            children.add(target[7:])
        modules[name] = sorted(children)
        todo.extend(sorted(children - modules.keys()))
    return {'modules': dict(sorted(modules.items())), 'external_imports': sorted(external),
            'dynamic_sites': sorted(dynamic, key=lambda x: (x['module'], x['line']))}


def compare(reference, candidate, *, expected_reference, expected_candidate, entries=ENTRIES):
    require(bool(entries) and len(set(entries)) == len(entries), 'distinct task entry modules required')
    old_root, old, old_sha = inventory(reference)
    new_root, new, new_sha = inventory(candidate)
    require(old_sha == expected_reference and new_sha == expected_candidate, 'pinned package fingerprint differs')
    before, after = dependency_graph(old, entries), dependency_graph(new, entries)
    closure = set(before['modules']) | set(after['modules'])
    changed = {}
    unchanged = []
    digest = lambda content: None if content is None else hashlib.sha256(content).hexdigest()
    for name in sorted(closure):
        left, right = old.get(name + '.py'), new.get(name + '.py')
        if left == right:
            unchanged.append(name)
            continue
        ot, nt = tree(left) if left is not None else None, tree(right) if right is not None else None
        od, nd = definitions(ot) if ot is not None else {}, definitions(nt) if nt is not None else {}
        changed[name] = {'reference_sha256': digest(left), 'candidate_sha256': digest(right),
                         'ast_identical': ot is not None and nt is not None and normalized(ot) == normalized(nt),
                         'changed_definitions': sorted(key for key in od.keys() | nd.keys() if od.get(key) != nd.get(key))}
    outside = sorted(name for name in old.keys() | new.keys()
                     if name.removesuffix('.py') not in closure and old.get(name) != new.get(name))
    # Verify the entire input snapshots again, including additions/removals.
    require(inventory(old_root)[1:] == (old, old_sha) and inventory(new_root)[1:] == (new, new_sha), 'source changed during comparison')
    return {'version': VERSION, 'reference_root': str(old_root), 'candidate_root': str(new_root),
            'reference_package_sha256': old_sha, 'candidate_package_sha256': new_sha,
            'entry_modules': list(entries), 'reference_dependencies': before, 'candidate_dependencies': after,
            'unchanged_dependency_modules': unchanged, 'changed_dependency_modules': changed,
            'changed_files_outside_dependency_closure': outside,
            'classification': 'dependency_changes_require_review' if changed else 'dependency_bytes_unchanged',
            'behavioral_compatibility_verified': False, 'a2_evidence_reusable': False, 'execution_authorized': False,
            'scope': 'Source and conservative import inventory only. Exact behavior/configuration, numerical runtime, trajectory/control and original-cohort checks remain separate; old evidence and source identities are never rewritten.'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for key in ('reference', 'candidate', 'expected-reference', 'expected-candidate', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).resolve()
        require(not output.exists(), 'preserve existing source inventory; choose a new output')
        require(all(not output.is_relative_to(Path(root).resolve()) for root in (args.reference, args.candidate)), 'output must be outside both input packages')
        result = compare(args.reference, args.candidate, expected_reference=args.expected_reference, expected_candidate=args.expected_candidate)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps({key: result[key] for key in ('classification', 'reference_package_sha256', 'candidate_package_sha256',
                                                     'changed_dependency_modules', 'a2_evidence_reusable')}, indent=2))
        return 0
    except (ValueError, OSError, SyntaxError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
