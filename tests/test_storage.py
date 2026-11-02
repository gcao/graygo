from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from collections import namedtuple
import os
import subprocess
import sys
from graygo.storage import CheckpointStore, DiskBudgetError, CorruptCheckpoint


def writer(value):
    def write(root):
        (root / "state.bin").write_bytes(value)
    return write


class StorageTests(TestCase):
    def test_rotation_permanent_archive_and_finalization(self):
        with TemporaryDirectory() as directory:
            store = CheckpointStore(directory, reserve_bytes=0)
            for step in range(4):
                store.save(step, writer(bytes([step])), metadata={"seed": 9}, estimated_peak_bytes=4096, kind="weights")
                store.save(step, writer(bytes([step])), metadata={"seed": 9}, estimated_peak_bytes=4096)
            self.assertEqual(len(list((Path(directory)/"weights").iterdir())), 4)
            self.assertEqual([int(p.name) for p in store._bundles()], [3,2])
            self.assertEqual(store.verify(store.latest())["step"], 3)
            store.finalize()
            self.assertEqual([int(p.name) for p in store._bundles()], [3])
            with self.assertRaises(FileExistsError):
                store.save(3, writer(b"x"), metadata={}, estimated_peak_bytes=0)

    def test_interrupted_writer_does_not_rotate_valid_state(self):
        with TemporaryDirectory() as directory:
            store = CheckpointStore(directory, reserve_bytes=0)
            for step in (1,2):
                store.save(step, writer(b"ok"), metadata={}, estimated_peak_bytes=4096)
            def fail(path):
                (path/"partial.bin").write_bytes(b"partial")
                raise OSError("simulated interrupted save")
            with self.assertRaises(OSError):
                store.save(3, fail, metadata={}, estimated_peak_bytes=4096)
            self.assertEqual([int(p.name) for p in store._bundles()], [2,1])
            self.assertEqual(list((Path(directory)/"staging").iterdir()), [])

    def test_process_death_and_recovery(self):
        with TemporaryDirectory() as directory:
            store = CheckpointStore(directory, reserve_bytes=0)
            store.save(1, writer(b"valid"), metadata={}, estimated_peak_bytes=4096)
            code = """
import os,sys
from graygo.storage import CheckpointStore
s=CheckpointStore(sys.argv[1],reserve_bytes=0)
def die(p):
 (p/'partial').write_bytes(b'partial')
 os._exit(73)
s.save(2,die,metadata={},estimated_peak_bytes=4096)
"""
            child = subprocess.run([sys.executable, "-c", code, directory], check=False)
            self.assertEqual(child.returncode, 73)
            self.assertEqual(store.verify(store.latest())["step"], 1)
            self.assertEqual(len(store.recover_staging()), 1)

    def test_checksum_corruption_falls_back_and_pointer_is_not_required(self):
        with TemporaryDirectory() as directory:
            store = CheckpointStore(directory, reserve_bytes=0)
            a = store.save(1, writer(b"first"), metadata={}, estimated_peak_bytes=4096)
            b = store.save(2, writer(b"second"), metadata={}, estimated_peak_bytes=4096)
            (Path(directory)/"resume"/"latest.json").unlink()
            self.assertEqual(store.latest(), b)
            (b/"state.bin").write_bytes(b"broken")
            with self.assertRaises(CorruptCheckpoint):
                store.verify(b)
            self.assertEqual(store.latest(), a)
            (a/"manifest.json").write_text('{}')
            with self.assertRaises(FileNotFoundError):
                store.latest()

    def test_projected_disk_guard_precedes_writer(self):
        Usage = namedtuple("Usage", "total used free")
        with TemporaryDirectory() as directory:
            store = CheckpointStore(directory, reserve_bytes=20)
            called = []
            with patch("graygo.storage.shutil.disk_usage", return_value=Usage(100,70,30)):
                with self.assertRaises(DiskBudgetError):
                    store.save(1, lambda p: called.append(p), metadata={}, estimated_peak_bytes=11)
            self.assertFalse(called)
            self.assertFalse(store._bundles())

    def test_symlink_payload_is_rejected_without_touching_target(self):
        with TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory)/"store", reserve_bytes=0)
            outside = Path(directory)/"outside"
            outside.write_bytes(b"keep")
            def link(path):
                (path/"link").symlink_to(outside)
            with self.assertRaises(ValueError):
                store.save(1, link, metadata={}, estimated_peak_bytes=4096)
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_crash_after_generation_commit_recovers_without_pointer(self):
        with TemporaryDirectory() as directory:
            store=CheckpointStore(directory,reserve_bytes=0)
            first=store.save(1,writer(b"first"),metadata={},estimated_peak_bytes=4096)
            with patch("graygo.storage.os.replace",side_effect=OSError("pointer commit interrupted")):
                with self.assertRaises(OSError):store.save(2,writer(b"second"),metadata={},estimated_peak_bytes=4096)
            self.assertEqual(store.verify(store.latest())["step"],2)
            self.assertEqual(store.verify(first)["step"],1)
            store.save(3,writer(b"third"),metadata={},estimated_peak_bytes=4096)
            self.assertEqual([int(p.name) for p in store._bundles()],[3,2])

    def test_live_writer_lock_blocks_cleanup(self):
        with TemporaryDirectory() as directory:
            store=CheckpointStore(directory,reserve_bytes=0)
            other=CheckpointStore(directory,reserve_bytes=0)
            with store._lock():
                with self.assertRaises(BlockingIOError):other.recover_staging()
