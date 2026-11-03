"""Finite old/current actor checks on sealed trained delayed-task weights.

This collects no training records or games and never resumes a trainer. Direct
actor API combinations are diagnostic probes, not a replay of training history.
"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch

VERSION = 'graygo/task-actor-boundary/1'
SEMANTIC = ('config', 'archive_source_sha256', 'archive_manifest_sha256', 'model_sha256',
            'adapter_sha256', 'observations_sha256', 'snapshot_sha256', 'raw_outputs_sha256',
            'scalar_commits_sha256', 'batched_commits_sha256', 'forced_commits_sha256',
            'restored_continuations_sha256', 'private_rng_sha256', 'counts', 'runtime',
            'torch_rng_sha256', 'numpy_global_rng_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class Transcript:
    """Typed, length-delimited exact values, including tensor dtype/shape/bits."""
    def __init__(self):
        self.h = hashlib.sha256()

    def add(self, value):
        def part(tag, data=b''):
            self.h.update(tag + str(len(data)).encode() + b':' + data)
        if value is None:
            part(b'null')
        elif isinstance(value, (bool, np.bool_)):
            part(b'bool', str(bool(value)).encode())
        elif isinstance(value, (int, np.integer)):
            part(b'int', str(int(value)).encode())
        elif isinstance(value, (float, np.floating)):
            part(b'float', float(value).hex().encode())
        elif isinstance(value, str):
            part(b'str', value.encode())
        elif isinstance(value, torch.Tensor):
            part(b'tensor')
            self.add(str(value.dtype)); self.add(tuple(value.shape))
            part(b'bits', value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, np.ndarray):
            part(b'array')
            self.add(value.dtype.str); self.add(tuple(value.shape)); self.add(bool(value.flags.writeable))
            part(b'bits', np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            require(all(isinstance(key, str) for key in value), 'transcript dictionaries require string keys')
            part(b'dict', str(len(value)).encode())
            for key in sorted(value):
                self.add(key); self.add(value[key])
        elif isinstance(value, (list, tuple)):
            part(b'list' if isinstance(value, list) else b'tuple', str(len(value)).encode())
            for item in value:
                self.add(item)
        elif is_dataclass(value) and not isinstance(value, type):
            part(b'dataclass', type(value).__name__.encode())
            self.add({field.name: getattr(value, field.name) for field in fields(value)})
        else:
            raise ValueError('unsupported transcript type: ' + str(type(value)))
        return self

    @property
    def sha256(self):
        return self.h.hexdigest()


def fingerprint(value):
    return Transcript().add(value).sha256


def verify_pair(reference, candidate, *, reference_source, candidate_source, archive_manifest_sha):
    for value, source in ((reference, reference_source), (candidate, candidate_source)):
        require(value.get('version') == VERSION and value.get('status') == 'complete', 'incomplete actor probe')
        require(value['source_sha256'] == source and value['archive_manifest_sha256'] == archive_manifest_sha, 'actor probe source/archive differs')
        require(value['source_and_archive_unchanged'] is True and value['global_rng_unchanged'] is True, 'actor probe mutated a source, archive or global RNG')
        require(value['new_training_records'] == value['new_evaluation_games'] == 0 and value['execution_authorized'] is False, 'actor probe became learning authority')
        require(value['runtime']['device'] == 'cpu' and value['runtime']['threads'] == 1, 'actor probe numerical mode differs')
        require(set(value['counts']) == {'observations', 'raw_forwards', 'scalar_commits', 'batched_commits', 'forced_commits', 'restored_continuations'}, 'actor probe work inventory differs')
        require(all(type(n) is int and n > 0 for n in value['counts'].values()), 'empty actor probe partition')
        require(value['counts']['observations'] > 128
                and value['counts']['raw_forwards'] == 2 * value['counts']['observations']
                and value['counts']['scalar_commits'] == 25 * value['counts']['observations']
                and value['counts']['batched_commits'] == 204
                and value['counts']['forced_commits'] == 36
                and value['counts']['restored_continuations'] == 150, 'partial actor probe inventory')
    for key in SEMANTIC:
        require(reference[key] == candidate[key], 'actor probe differs: ' + key)
    return {'version': VERSION, 'status': 'finite_actor_boundary_verified',
            'reference_source_sha256': reference_source, 'candidate_source_sha256': candidate_source,
            'archive_manifest_sha256': archive_manifest_sha, 'compared_fields': list(SEMANTIC),
            'counts_per_package': reference['counts'], 'task': reference['config']['task'], 'learner': reference['config']['learner'],
            'complete_training_trajectory_verified': False, 'source_compatibility_qualified': False,
            'a2_evidence_reusable': False, 'execution_authorized': False,
            'scope': 'Exact finite trained-weight actor outputs, labels, private draws, batching and restore boundary only; no old/current learning trajectory, rule-engine or complete A2/project-entry qualification.'}


def probe(archive, *, expected_source, expected_archive_source, expected_archive_manifest, time_cap_seconds):
    from graygo.manifest import source_digest
    from graygo.storage import CheckpointStore
    from graygo.task_adapter import TaskAdapter
    from graygo.task_training import NFSPTaskConfig
    from graygo.nfsp_collection import FrozenPolicies, EpisodeActor, commit_many
    from graygo.schedules import EpisodeChoice
    started = time.monotonic()
    require(type(time_cap_seconds) in (int, float) and 0 < time_cap_seconds <= 600, 'bounded positive actor probe time required')

    def check_time():
        require(time.monotonic() - started <= time_cap_seconds, 'actor probe time ceiling exhausted')

    archive = Path(archive).resolve()
    require(source_digest() == expected_source and file_sha(archive / 'manifest.json') == expected_archive_manifest, 'pinned actor input differs')
    seal = CheckpointStore.verify(archive)
    require(seal['kind'] == 'weights' and seal['step'] == 15000000 and seal['metadata']['source_sha256'] == expected_archive_source, 'original full trained archive required')
    hashes = {str(path): file_sha(path) for path in archive.rglob('*') if path.is_file()}
    payload = torch.load(archive / 'models.pt', map_location='cpu', weights_only=True)
    config = NFSPTaskConfig(**payload['config'])
    require(config.formal_protocol() and config.task == 'delayed' and config.device == 'cpu' and config.q_reuse == 4,
            'probe requires the original primary CPU delayed R4 recipe')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1729)
    np.random.seed(1729)
    adapter = TaskAdapter(config)
    q, average = adapter.models()
    q.load_state_dict(payload['q']); average.load_state_dict(payload['average'])
    q.eval().requires_grad_(False); average.eval().requires_grad_(False)
    before_torch, before_numpy = torch.get_rng_state().clone(), np.random.get_state()
    snapshot = FrozenPolicies(q, average, version=seal['step'], average_ready=True)
    saved = snapshot.state_dict()
    restored_snapshot = FrozenPolicies(q, average, version=seal['step'], average_ready=True)
    restored_snapshot.load_state_dict(saved)
    require(fingerprint(saved) == fingerprint(restored_snapshot.state_dict()), 'default snapshot restore differs')
    require(getattr(snapshot, 'forward_pool', None) is None, 'A2 default actor unexpectedly selected graph execution')
    observations = adapter.observations
    require(len(observations) == 2 * len(adapter.states) and len(observations) > 100, 'complete both-role delayed observation inventory required')
    transcripts = {name: Transcript() for name in ('observations', 'raw_outputs', 'scalar_commits', 'batched_commits', 'forced_commits', 'restored_continuations', 'private_rng')}
    counts = {'observations': len(observations), 'raw_forwards': 0, 'scalar_commits': 0, 'batched_commits': 0, 'forced_commits': 0, 'restored_continuations': 0}
    with torch.no_grad():
        for index, obs in enumerate(observations):
            if index % 64 == 0:
                check_time()
            transcripts['observations'].add((index, obs.spatial, obs.global_features, obs.legal))
            inputs = {'spatial': torch.tensor(obs.spatial[None], dtype=torch.float32),
                      'globals': torch.tensor(obs.global_features[None], dtype=torch.float32)}
            transcripts['raw_outputs'].add((index, snapshot.q(**inputs), snapshot.average(**inputs)))
            counts['raw_forwards'] += 2
    schedule = config.schedule()
    variants = [(True, 'q', False), (True, 'q', True), (False, 'q', False), (False, 'average', False), (False, 'uniform', False)]
    records = (0, 5000000, 6000000, 7000000, 15000000)
    resume_indices = {0, 1, 17, 63, 127, len(observations) - 1}
    for look, record in enumerate(records):
        for variant, (best, kind, labels_enabled) in enumerate(variants):
            choice = EpisodeChoice(record, best, kind, schedule.epsilon(record), schedule.ramp(record), .05 if best else .5, .25)
            eligible = schedule.label_eligible(choice, record, opening_enabled=labels_enabled)
            for index, obs in enumerate(observations):
                if index % 64 == 0:
                    check_time()
                actor = EpisodeActor(snapshot, choice, np.random.default_rng(1000000 * look + 100000 * variant + index))
                commitment = actor.commit(obs, label_eligible=eligible)
                transcripts['scalar_commits'].add((look, variant, index, commitment))
                transcripts['private_rng'].add((look, variant, index, actor.state_dict()))
                counts['scalar_commits'] += 1
                if index in resume_indices:
                    actor_state = actor.state_dict()
                    restored = EpisodeActor.restore(restored_snapshot, actor_state)
                    following = observations[(index + 1) % len(observations)]
                    first = actor.commit(following, label_eligible=eligible)
                    second = restored.commit(following, label_eligible=eligible)
                    require(fingerprint(first) == fingerprint(second) and fingerprint(actor.state_dict()) == fingerprint(restored.state_dict()), 'private actor restore continuation differs')
                    transcripts['restored_continuations'].add((look, variant, index, second, restored.state_dict()))
                    counts['restored_continuations'] += 1
    # Grouped source path and immutable-version separation, independent of the scalar path.
    alternate = FrozenPolicies(q, average, version=seal['step'] + 1, average_ready=True)
    for layout in ('q', 'average', 'uniform', 'mixed'):
        for batch in (1, 2, 16, 32):
            requests = []
            for index in range(batch):
                kind = ('q', 'average', 'uniform')[index % 3] if layout == 'mixed' else layout
                best = kind == 'q'
                choice = EpisodeChoice(15000000, best, kind, schedule.epsilon(15000000), 1., .05 if best else .5, .25)
                frozen = alternate if layout == 'mixed' and index % 2 else snapshot
                actor = EpisodeActor(frozen, choice, np.random.default_rng(9000000 + batch * 100 + index))
                requests.append((actor, observations[(17 * index + batch) % len(observations)], best))
            commits = commit_many(requests)
            transcripts['batched_commits'].add((layout, batch, commits, [actor.state_dict() for actor, _, _ in requests]))
            counts['batched_commits'] += len(commits)
    for root_index, root in enumerate(adapter.roots):
        obs = adapter.game.observation(root, root.controller)
        for kind in ('q', 'average', 'uniform'):
            choice = EpisodeChoice(15000000, kind == 'q', kind, .01, 1., .05 if kind == 'q' else .5, .25)
            for action in adapter.game.actions(root, root.controller):
                actor = EpisodeActor(snapshot, choice, np.random.default_rng(8000000 + root_index * 100 + action))
                before = fingerprint(actor.state_dict())
                commitment = actor.commit(obs, forced_action=action, supplemental=True)
                require(before == fingerprint(actor.state_dict()) and commitment.probability == 1. and commitment.forced, 'forced root consumed a private draw or changed its probability')
                transcripts['forced_commits'].add((root_index, kind, action, commitment))
                counts['forced_commits'] += 1
    check_time()
    require(torch.equal(before_torch, torch.get_rng_state()) and fingerprint(before_numpy) == fingerprint(np.random.get_state()), 'probe changed global RNG state')
    require(source_digest() == expected_source and hashes == {str(path): file_sha(path) for path in archive.rglob('*') if path.is_file()}, 'actor source/archive changed during probe')
    result = {'version': VERSION, 'status': 'complete', 'source_sha256': expected_source,
              'archive_source_sha256': expected_archive_source, 'archive_manifest_sha256': expected_archive_manifest,
              'model_sha256': file_sha(archive / 'models.pt'), 'config': config.plan(),
              'adapter_sha256': fingerprint(adapter.manifest()), 'snapshot_sha256': fingerprint(saved),
              'counts': counts, 'runtime': {'device': 'cpu', 'threads': torch.get_num_threads(), 'deterministic': torch.are_deterministic_algorithms_enabled(),
                                          'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)},
              'torch_rng_sha256': fingerprint(before_torch), 'numpy_global_rng_sha256': fingerprint(before_numpy),
              'source_and_archive_unchanged': True, 'global_rng_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
              'new_training_records': 0, 'new_evaluation_games': 0, 'execution_authorized': False,
              'scope': 'Finite direct actor API probes on original trained weights; no trainer, optimizer updates, new records, collected games or claim of original-history replay.'}
    result.update({name + '_sha256': transcript.sha256 for name, transcript in transcripts.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for key in ('archive', 'expected-source-sha', 'expected-archive-source-sha', 'expected-archive-manifest-sha', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--time-cap-seconds', type=float, default=270.)
    args = parser.parse_args(argv)
    try:
        out = Path(args.output).resolve()
        archive = Path(args.archive).resolve()
        protected = archive.parent.parent if archive.parent.name == 'weights' else archive
        require(not out.exists() and not out.is_relative_to(protected), 'new probe output outside the original training inputs required')
        result = probe(args.archive, expected_source=args.expected_source_sha, expected_archive_source=args.expected_archive_source_sha,
                       expected_archive_manifest=args.expected_archive_manifest_sha, time_cap_seconds=args.time_cap_seconds)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        print(json.dumps({'status': result['status'], 'learner': result['config']['learner'], 'counts': result['counts'], 'elapsed_seconds': result['elapsed_seconds']}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
