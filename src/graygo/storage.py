"""Checksummed atomic checkpoints with bounded heavy-state retention (§7)."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid

FORMAT_VERSION = 1
DEFAULT_RESERVE = 20 * 1024**3
GENERATION = re.compile(r"[0-9]{12,}")


class DiskBudgetError(RuntimeError):
    pass


class CorruptCheckpoint(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CheckpointStore:
    def __init__(self, root, *, reserve_bytes=DEFAULT_RESERVE):
        if reserve_bytes < 0:
            raise ValueError("negative disk reserve")
        self.root = Path(root).resolve()
        self.reserve_bytes = reserve_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("weights", "resume", "staging"):
            path = self.root / name
            if path.is_symlink():
                raise ValueError("managed checkpoint directories cannot be symlinks")
            path.mkdir(exist_ok=True)

    def ensure_space(self, estimated_peak_bytes=0):
        if estimated_peak_bytes < 0:
            raise ValueError("negative peak-write estimate")
        free = shutil.disk_usage(self.root).free
        if free - estimated_peak_bytes < self.reserve_bytes:
            raise DiskBudgetError(f"checkpoint paused: free={free}, peak_write={estimated_peak_bytes}, reserve={self.reserve_bytes}")
        return free

    @contextmanager
    def _lock(self):
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def verify(path):
        path = Path(path)
        try:
            if path.is_symlink() or not path.is_dir():
                raise ValueError("invalid bundle directory")
            manifest_file = path / "manifest.json"
            seal = path / "manifest.sha256"
            if manifest_file.is_symlink() or seal.is_symlink():
                raise ValueError("symlink in bundle")
            if digest(manifest_file) != seal.read_text().strip():
                raise ValueError("manifest checksum mismatch")
            manifest = json.loads(manifest_file.read_text())
            if not isinstance(manifest, dict) or manifest.get("kind") not in ("resume", "weights"):
                raise ValueError("invalid bundle manifest")
            if type(manifest.get("step")) is not int or manifest["step"] < 0 or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
                raise ValueError("invalid manifest generation/inventory")
            if GENERATION.fullmatch(path.name) and int(path.name) != manifest["step"]:
                raise ValueError("generation disagrees with manifest")
            if path.parent.name in ("resume", "weights") and path.parent.name != manifest["kind"]:
                raise ValueError("bundle kind disagrees with destination")
            if manifest["format_version"] != FORMAT_VERSION:
                raise ValueError("unsupported bundle format")
            expected = set(manifest["files"]) | {"manifest.json", "manifest.sha256"}
            actual = set()
            for file in path.rglob("*"):
                if file.is_symlink():
                    raise ValueError("symlink in bundle")
                if file.is_file():
                    actual.add(file.relative_to(path).as_posix())
            if actual != expected:
                raise ValueError("bundle file inventory mismatch")
            for name, entry in manifest["files"].items():
                parts = PurePosixPath(name)
                if parts.is_absolute() or ".." in parts.parts or parts.as_posix() != name:
                    raise ValueError("invalid payload path")
                file = path / name
                if file.stat().st_size != entry["bytes"] or digest(file) != entry["sha256"]:
                    raise ValueError(f"payload checksum mismatch: {name}")
            return manifest
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise CorruptCheckpoint(f"invalid checkpoint {path}: {exc}") from exc

    def _bundles(self):
        return sorted((p for p in (self.root / "resume").iterdir()
                       if GENERATION.fullmatch(p.name) and p.is_dir() and not p.is_symlink()),
                      key=lambda p: int(p.name), reverse=True)

    def latest(self):
        # The pointer is advisory: recover a fully committed generation if a crash
        # occurred after directory commit but before pointer publication.
        for path in self._bundles():
            try:
                self.verify(path)
                return path
            except CorruptCheckpoint:
                continue
        raise FileNotFoundError("no valid resume checkpoint")

    def save(self, step, writer, *, metadata, estimated_peak_bytes, kind="resume"):
        if type(step) is not int or step < 0 or kind not in ("resume", "weights"):
            raise ValueError("invalid step or checkpoint kind")
        # Fail on invalid metadata before invoking the state writer.
        json.dumps(metadata, allow_nan=False)
        with self._lock():
            self.ensure_space(estimated_peak_bytes)
            target = self.root / kind / f"{step:012d}"
            if target.exists():
                raise FileExistsError(target)
            if kind == "resume" and self._bundles() and step <= int(self._bundles()[0].name):
                raise ValueError("resume generations must increase monotonically")
            stage = self.root / "staging" / uuid.uuid4().hex
            stage.mkdir()
            try:
                writer(stage)
                files = {}
                for path in sorted(stage.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("payload symlinks are forbidden")
                    if path.is_file():
                        name = path.relative_to(stage).as_posix()
                        if name in ("manifest.json", "manifest.sha256"):
                            raise ValueError("writer used a reserved filename")
                        files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
                        with path.open("rb") as data:
                            os.fsync(data.fileno())
                if not files:
                    raise ValueError("empty checkpoint payload")
                manifest = {"format_version": FORMAT_VERSION, "kind": kind, "step": step,
                            "metadata": metadata, "files": files}
                _json(stage / "manifest.json", manifest)
                with (stage / "manifest.sha256").open("x") as seal:
                    seal.write(digest(stage / "manifest.json") + "\n")
                    seal.flush()
                    os.fsync(seal.fileno())
                for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
                    _sync_dir(directory)
                _sync_dir(stage)
                self.verify(stage)
                self.ensure_space()  # detect an underestimated payload before commit
                stage.rename(target)
                _sync_dir(target.parent)
                self.verify(target)
                if kind == "resume":
                    pointer = self.root / "resume" / (".latest-" + uuid.uuid4().hex)
                    _json(pointer, {"generation": target.name, "manifest_sha256": digest(target / "manifest.json")})
                    os.replace(pointer, self.root / "resume" / "latest.json")
                    _sync_dir(target.parent)
                    self._prune(2)
                return target
            finally:
                if stage.exists():
                    shutil.rmtree(stage)

    def _prune(self, keep):
        valid = []
        for path in self._bundles():
            try:
                self.verify(path)
                valid.append(path)
            except CorruptCheckpoint:
                continue
        # Never discard old state unless enough newer verified bundles exist.
        protected = set(valid[:keep])
        if len(protected) < keep:
            return
        for path in self._bundles():
            if path not in protected:
                shutil.rmtree(path)
        _sync_dir(self.root / "resume")

    def finalize(self):
        with self._lock():
            latest = self.latest()
            self.verify(latest)
            self._prune(1)
            return latest

    def recover_staging(self):
        """Exclusive lock proves no writer in this store currently owns staging."""
        removed = []
        with self._lock():
            for path in (self.root / "staging").iterdir():
                if re.fullmatch(r"[a-f0-9]{32}", path.name) and path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    removed.append(path.name)
        return removed
