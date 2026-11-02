"""Locked, cumulatively budgeted execution around the board milestone journal.

This is the execution mechanism, not the project A2/resource/cohort authority.
The public CLI exposes only bounded diagnostics. A formal scope controller must
verify its evidence and restrict the endpoint before using this component.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .board_journal import BoardMilestoneJournal, OpeningNoGo, OpeningPending
from .board_runtime import BoardNFSPRuntime, configure_runtime
from .confidence import canonical_sha
from .evaluation import EmptyStarts
from .manifest import _command, source_digest, write_report
from .rule_state import RuleStateSpec
from .storage import CheckpointStore, DEFAULT_RESERVE, DiskBudgetError, digest

EXECUTION_VERSION = 'graygo/board-execution/1'


@dataclass(frozen=True)
class BoardExecutionPlan:
    time_ceiling_seconds: float
    checkpoint_every: int = 102400
    heartbeat_seconds: float = 30.
    checkpoint_headroom_bytes: int = 1024**3

    def __post_init__(self):
        for value in (self.time_ceiling_seconds, self.heartbeat_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('finite positive cumulative time/heartbeat limits required')
        if self.heartbeat_seconds > 60:
            raise ValueError('execution clock heartbeat must be at most sixty seconds')
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 2 or self.checkpoint_every % 2:
            raise ValueError('even fresh-record checkpoint cadence required')
        if type(self.checkpoint_headroom_bytes) is not int or self.checkpoint_headroom_bytes < 65536:
            raise ValueError('projected checkpoint write headroom required')

    def to_dict(self):
        return {'version': EXECUTION_VERSION, **asdict(self)}


def _runtime_identity(config):
    torch.set_num_threads(1)
    settings = configure_runtime(config.device, config.deterministic)
    device = torch.device(config.device)
    gpu = None
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        gpu = {'name': properties.name, 'total_memory': properties.total_memory,
               'capability': [properties.major, properties.minor]}
    cpu = None
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), None)
    driver = _command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']) if gpu else None
    return {'settings': settings, 'gpu': gpu, 'driver': driver, 'cpu': cpu, 'python': platform.python_version(),
            'numpy': np.__version__,
            'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if device.type == 'cuda' else None,
            'machine': platform.machine()}


class BoardExecutor:
    """Hold the execution lock from initialization through final clock commit.

    `run(until_records)` is an operational boundary, never a funding grant or a
    cohort decision. Missing opening evidence stops collection. Only the outer
    scope controller can select formal work or continue a passed pilot cohort.
    """
    def __init__(self, root, config, journal_plan, execution_plan, *, starts=None,
                 resume=False, reserve_bytes=DEFAULT_RESERVE, monotonic=time.monotonic,
                 wall_time=time.time, prior_setup_seconds=0.):
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError('nonnegative execution disk reserve required')
        if 0 not in journal_plan.milestones:
            raise ValueError('executor requires a permanent initialized model at record zero')
        if type(prior_setup_seconds) not in (int, float) or not math.isfinite(prior_setup_seconds) or prior_setup_seconds < 0:
            raise ValueError('nonnegative measured pre-entry setup time required')
        if journal_plan.scope == 'diagnostic' and execution_plan.time_ceiling_seconds > 3600:
            raise ValueError('diagnostic execution cannot allocate more than one cumulative hour')
        self.root = Path(root).resolve()
        self.config, self.journal_plan, self.plan = config, journal_plan, execution_plan
        self.starts = starts or EmptyStarts()
        self.resume, self.reserve = resume, reserve_bytes
        self.monotonic, self.wall_time = monotonic, wall_time
        self.prior_setup_seconds = prior_setup_seconds
        self.lock = self.runtime = self.journal = self.clock = None
        self.active = False
        self.used = False
        self.clock_persisted = False

    @property
    def clock_path(self):
        return self.root / 'execution.json'

    def _write_clock(self):
        body = {'version': EXECUTION_VERSION, 'contract': self.contract, 'clock': self.clock}
        write_report(self.clock_path, {'body': body, 'sha256': canonical_sha(body)}, reserve_bytes=self.reserve)
        # Persist the rename as well as the report's fsynced payload.
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.clock_persisted = True

    def _read_clock(self):
        wrapper = json.loads(self.clock_path.read_text())
        body = wrapper.get('body')
        if wrapper.get('sha256') != canonical_sha(body) or body.get('version') != EXECUTION_VERSION:
            raise ValueError('execution clock checksum/version differs')
        if body.get('contract') != self.contract:
            raise ValueError('execution source, plan, runtime, decoder or start distribution changed')
        clock = body['clock']
        required = {'observed_seconds', 'recovery_charge_seconds', 'inflight_since', 'last_checkpoint',
                    'training_run_sha256', 'sessions', 'unclean_recoveries', 'status'}
        if not isinstance(clock, dict) or set(clock) != required:
            raise ValueError('execution clock field inventory differs')
        for key in ('observed_seconds', 'recovery_charge_seconds'):
            if type(clock.get(key)) not in (int, float) or not math.isfinite(clock[key]) or clock[key] < 0:
                raise ValueError('invalid cumulative execution clock')
        if clock.get('inflight_since') is not None:
            value = clock['inflight_since']
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid in-flight execution clock')
        for key in ('sessions', 'unclean_recoveries'):
            if type(clock[key]) is not int or clock[key] < 0:
                raise ValueError('invalid execution session/recovery count')
        fingerprint = clock['training_run_sha256']
        if fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64
                                        or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('invalid execution training-run binding')
        checkpoint = clock['last_checkpoint']
        if checkpoint is not None:
            if (not isinstance(checkpoint, dict) or set(checkpoint) != {'records', 'manifest_sha256'}
                    or type(checkpoint['records']) is not int or checkpoint['records'] % 2
                    or not 0 <= checkpoint['records'] <= self.journal_plan.records
                    or not isinstance(checkpoint['manifest_sha256'], str) or len(checkpoint['manifest_sha256']) != 64):
                raise ValueError('invalid durable checkpoint pointer')
        return clock

    def _flush(self, status=None, *, closing=False):
        self.clock['observed_seconds'] = self.base_observed + max(0., self.monotonic() - self.session_started)
        self.clock['inflight_since'] = None if closing else self.wall_time()
        if status is not None:
            self.clock['status'] = status
        self._write_clock()
        self.last_heartbeat = self.monotonic()

    def _checkpoint_headroom(self):
        # Use full-capacity buffers and the maximum number of distinct frozen
        # episode models, including future Adam moments. Computing this once
        # avoids serializing or walking every tensor before each collector chunk.
        q = list(self.runtime.backend.q.state_dict().values())
        average = list(self.runtime.backend.average.state_dict().values())
        q_bytes = sum(t.numel() * t.element_size() for t in q)
        average_bytes = sum(t.numel() * t.element_size() for t in average)
        buffers = sum(a.nbytes for b in (self.runtime.replay, self.runtime.reservoir) for a in b.arrays.values())
        tensors = 4 * q_bytes + 3 * average_bytes + (self.config.envs + 1) * (q_bytes + average_bytes)
        metadata = 1024**2 + (self.config.envs + 8) * (len(q) + len(average)) * 2048
        return max(self.plan.checkpoint_headroom_bytes, math.ceil(1.10 * (buffers + tensors + metadata)))

    def __enter__(self):
        if self.used:
            raise RuntimeError('create a fresh executor context for each invocation')
        self.used = True
        self.session_started, session_wall = self.monotonic(), self.wall_time()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.execution.lock').open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.store = CheckpointStore(self.root, reserve_bytes=self.reserve)
            self.store.ensure_space(self.plan.checkpoint_headroom_bytes)
            self.contract = {'version': EXECUTION_VERSION, 'runtime': self.config.plan(),
                             'journal': self.journal_plan.to_dict(), 'execution': self.plan.to_dict(),
                             'training_starts': self.starts.identity, 'source_sha256': source_digest(),
                             'native_schema': RuleStateSpec(self.config.rules, self.config.native_library).to_dict(),
                             'numerical_runtime': _runtime_identity(self.config), 'reserve_bytes': self.reserve,
                             'authority': 'execution mechanics only; no A2/resource/cohort grant'}
            if self.clock_path.exists():
                if not self.resume:
                    raise FileExistsError('executor already exists; inspect its live process and resume explicitly')
                self.clock = self._read_clock()
            else:
                if (self.resume or (self.root / 'run.json').exists() or list((self.root / 'resume').glob('[0-9]*'))
                        or list((self.root / 'weights').glob('[0-9]*'))):
                    raise ValueError('original execution clock is missing; cannot reset an existing allocation')
                self.clock = {'observed_seconds': 0., 'recovery_charge_seconds': 0., 'inflight_since': None,
                              'last_checkpoint': None, 'training_run_sha256': None, 'sessions': 0,
                              'unclean_recoveries': 0, 'status': 'initializing'}
            previous = self.clock['inflight_since']
            if previous is not None:
                remaining = max(0., self.plan.time_ceiling_seconds - self.clock['observed_seconds']
                                - self.clock['recovery_charge_seconds'] - self.prior_setup_seconds)
                # An unclean interval includes offline time conservatively. Clock
                # reversal cannot purchase a fresh remaining allocation.
                gap = session_wall - self.prior_setup_seconds - previous
                self.clock['recovery_charge_seconds'] += min(remaining, gap) if gap >= 0 else remaining
                self.clock['unclean_recoveries'] += 1
            self.base_observed = self.clock['observed_seconds'] + self.prior_setup_seconds
            self.clock['sessions'] += 1
            self._flush('initializing')
            self.runtime = BoardNFSPRuntime(self.config, starts=self.starts)
            self.write_headroom = self._checkpoint_headroom()
            self.store.ensure_space(self.write_headroom)
            manifest_exists = (self.root / 'run.json').exists()
            if self.clock['training_run_sha256'] is not None and not manifest_exists:
                raise ValueError('initialized training manifest is missing; cannot create a replacement')
            self.journal = BoardMilestoneJournal(self.root, self.runtime, self.journal_plan,
                                                resume=manifest_exists, reserve_bytes=self.reserve)
            if self.clock['training_run_sha256'] not in (None, self.journal.run_sha256):
                raise ValueError('execution clock belongs to another initialized training run')
            self.clock['training_run_sha256'] = self.journal.run_sha256
            try:
                latest = self.store.latest()
            except FileNotFoundError:
                latest = None
            durable = self.clock['last_checkpoint']
            if durable is not None:
                recorded = self.root / 'resume' / f'{durable["records"]:012d}'
                self.store.verify(recorded)
                if digest(recorded / 'manifest.json') != durable['manifest_sha256']:
                    raise ValueError('recorded durable checkpoint changed')
                if latest is None or int(latest.name) < durable['records']:
                    raise ValueError('execution cannot silently roll back its durable record clock')
            if latest is not None:
                self.journal.restore(latest)
                # A bundle may commit just before a crash in the separate clock
                # write. Its verified journal is adopted without reinitializing.
                self._bind_checkpoint(latest)
            else:
                if durable is not None:
                    raise ValueError('durable execution checkpoint is missing')
                # Recover initialization interrupted before the first bundle.
                self._checkpoint()
            self.active = True
            self._flush('ready')
            return self
        except BaseException:
            # A rejected contract/clock must remain byte-for-byte untouched.
            if self.clock is not None and hasattr(self, 'base_observed'):
                try:
                    self._flush('failed', closing=True)
                except (OSError, RuntimeError):
                    pass
            self._unlock()
            raise

    def _unlock(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None
        self.active = False

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.clock is not None:
                try:
                    self._flush('failed' if exc_type else None, closing=True)
                except (OSError, RuntimeError):
                    # Preserve the earlier in-flight clock; recovery will charge
                    # the unrecorded interval. Never lower the disk reserve.
                    self.clock_persisted = False
        finally:
            self._unlock()

    def _bind_checkpoint(self, path):
        seal = self.store.verify(path)
        self.clock['last_checkpoint'] = {'records': seal['step'], 'manifest_sha256': digest(path / 'manifest.json')}
        self._flush()

    def _checkpoint(self):
        path = self.journal.checkpoint()
        self._bind_checkpoint(path)
        return path

    def _spent(self):
        return self.base_observed + max(0., self.monotonic() - self.session_started) + self.clock['recovery_charge_seconds']

    def report(self, status):
        return {'version': EXECUTION_VERSION, 'status': status, 'records': self.runtime.records,
                'last_durable_checkpoint': deepcopy(self.clock['last_checkpoint']),
                'training_run_sha256': self.journal.run_sha256, 'contract_sha256': canonical_sha(self.contract),
                'clock': deepcopy(self.clock), 'clock_persisted': self.clock_persisted,
                'counters': asdict(self.runtime.backend.counters), 'metrics': deepcopy(self.runtime.collector.metrics),
                'reservoir_seen': self.runtime.reservoir.seen, 'archived_records': sorted(int(n) for n in self.journal.archived),
                'pre_collection_checkpoint_headroom_bytes': self.write_headroom,
                'scope': self.journal_plan.scope, 'a2_complete': False, 'formal_scope_approved': False,
                'authority': 'mechanism only; formal entry and cohort verification are separate',
                'budget_boundary': 'between paired collection chunks; initialization, one chunk and safe IO may overrun',
                'unclean_charge': 'unobserved in-flight interval includes offline time, capped at remaining allocation'}

    def run(self, until_records=None, *, stop_requested=lambda: False):
        if not self.active:
            raise RuntimeError('hold the execution context/lock while running')
        until = self.journal_plan.records if until_records is None else until_records
        if type(until) is not int or until % 2 or not self.runtime.records <= until <= self.journal_plan.records:
            raise ValueError('paired operational endpoint within the unchanged run horizon required')
        status = 'complete' if until == self.journal_plan.records else 'checkpoint_boundary'
        try:
            while self.runtime.records < until:
                if stop_requested():
                    status = 'interrupted'
                    break
                if self._spent() >= self.plan.time_ceiling_seconds:
                    status = 'budget_limited'
                    break
                # The next full checkpoint's frozen headroom is checked before
                # collection; save additionally checks its actual peak estimate.
                self.store.ensure_space(self.write_headroom)
                n = self.runtime.records
                regular = (n // self.plan.checkpoint_every + 1) * self.plan.checkpoint_every
                count = min(2 * self.config.envs, self.runtime.backend.counters.until_q_boundary(self.runtime.backend.config),
                            until - n, self.journal.next_boundary() - n, regular - n)
                try:
                    self.journal.before_advance(count)
                except OpeningPending:
                    status = 'opening_pending'
                    break
                except OpeningNoGo:
                    status = 'opening_no_go'
                    break
                self.runtime.advance(count, opening_enabled=self.journal.label_insertion_allowed())
                n = self.runtime.records
                if n in self.journal_plan.milestones or n % self.plan.checkpoint_every == 0:
                    self._checkpoint()
                elif self.monotonic() - self.last_heartbeat >= self.plan.heartbeat_seconds:
                    self._flush('running')
            self._checkpoint()
            if self.runtime.records == self.journal_plan.records or status == 'opening_no_go':
                # Publish the new durable pointer before pruning old bundles.
                self.journal.finalize()
            self._flush(status)
        except DiskBudgetError as error:
            status = 'storage_limited'
            self.clock_persisted = False
            try:
                self._flush(status)
            except DiskBudgetError:
                pass
            result = self.report(status)
            result['storage_error'] = str(error)
            result['unsaved_records'] = self.runtime.records - (self.clock['last_checkpoint'] or {'records': 0})['records']
            return result
        return self.report(status)
