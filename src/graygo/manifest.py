"""Explicit run metadata: no ambient environment-variable or credential dumps."""
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import uuid
from .engine import RULE_VERSION
from .encoding import ENCODING_VERSION
from .agents import BASELINE_VERSION
from .storage import DEFAULT_RESERVE, DiskBudgetError


def _command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_digest():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted([*root.glob("*.py"),*(root/"native").glob("*.cpp")]):
        h.update(path.relative_to(root).as_posix().encode()+b"\0"+path.read_bytes()+b"\0")
    return h.hexdigest()


def make_manifest(*, seed, command, rules=None):
    versions = {}
    for package in ("numpy", "scipy", "torch"):
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: versions[package] = None
    cpu = None
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), None)
    design = Path(__file__).resolve().parents[2] / "docs" / "design.md"
    design_sha = hashlib.sha256(design.read_bytes()).hexdigest() if design.exists() else None
    host_memory=None
    if Path("/proc/meminfo").exists():
        memory={line.split(":",1)[0]:int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemTotal:","MemAvailable:","SwapTotal:"))}
        host_memory={"total_bytes":memory.get("MemTotal"),"available_bytes":memory.get("MemAvailable"),"swap_bytes":memory.get("SwapTotal")}
    return {"format_version": 1, "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "command": command,
            "rule_version": "graygo/3" if rules and rules.get("scoring")=="iterative" else "graygo/2" if rules and rules.get("ko")=="full-history" else RULE_VERSION, "encoding_version": ENCODING_VERSION, "baseline_version": BASELINE_VERSION,
            "rules": rules, "source_sha256": source_digest(), "design_sha256": design_sha,
            "git_commit": _command(["git","rev-parse","HEAD"]),
            "git_dirty": bool(_command(["git","status","--porcelain"])),
            "hostname": socket.gethostname(), "platform": platform.platform(), "cpu": cpu,
            "cpu_threads": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
            "host_memory":host_memory,"filesystem":_command(["stat","-f","-c","%T",str(Path.cwd())]), "python": sys.version, "dependencies": versions,
            "gpu": _command(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"]),
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free}


def write_report(path, report, *, reserve_bytes=DEFAULT_RESERVE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    if shutil.disk_usage(path.parent).free-len(data)-65536 < reserve_bytes:
        raise DiskBudgetError("report would violate the destination disk reserve")
    temp = path.parent / ("."+path.name+"-"+uuid.uuid4().hex)
    try:
        with temp.open("xb") as out:
            out.write(data); out.flush(); os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)
