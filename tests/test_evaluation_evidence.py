from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from graygo.confidence import ConfidenceFamily
from graygo.engine import Rules,State,resolve
from graygo.evaluation import BaselinePolicy,EpisodeMixture,EvaluationPlan,EvaluationRun,FixedStarts
from graygo.evaluation_evidence import FrozenEvaluationEvidence,evaluation_read_lock


def files(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob('*') if p.is_file()}


def alter_unit(root,change,index=0):
    db=sqlite3.connect(Path(root)/'evidence.sqlite')
    try:
        payload=db.execute('SELECT payload FROM units WHERE idx=?',(index,)).fetchone()[0]
        row=json.loads(payload);change(row);payload=json.dumps(row,sort_keys=True,separators=(',',':'))
        with db:db.execute('UPDATE units SET payload=?,sha256=? WHERE idx=?',(payload,hashlib.sha256(payload.encode()).hexdigest(),index))
    finally:db.close()


class ReadOnlyEvaluationTests(unittest.TestCase):
    def create(self,path,*,mixture=False,incumbent=False,openings=None):
        family=ConfidenceFamily('read-only',('endpoint',),(4,8))
        plan=EvaluationPlan('read-only','subject','endpoint',Rules(3,2),91,family.looks,family.sha256,
                            'improvement' if incumbent else 'utility')
        subject=BaselinePolicy('random')
        if mixture:subject=EpisodeMixture((subject,BaselinePolicy('tactical')),(.25,.75))
        run=EvaluationRun(plan,path,subject,BaselinePolicy('tactical'),openings=openings,
                          incumbent=BaselinePolicy('random') if incumbent else None,reserve_bytes=0)
        run.run(4,time_cap_seconds=30)
        return run,family

    def test_real_scored_utility_snapshot_matches_collector_without_any_input_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'evidence with # and spaces';run,family=self.create(root)
            expected=run.estimate(family,4);before=files(root)
            snapshot=FrozenEvaluationEvidence.load(root)
            self.assertEqual(snapshot.estimate(family,4),expected)
            self.assertEqual(snapshot.available_games,4);self.assertEqual(snapshot.largest_completed_look,4)
            self.assertEqual(files(root),before)

    def test_read_only_database_and_input_tree_work_without_side_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,family=self.create(root);expected=run.estimate(family,4);before=files(root)
            (root/'evidence.sqlite').chmod(0o400);(root/'plan.json').chmod(0o400);root.chmod(0o500)
            try:
                with evaluation_read_lock(root):
                    self.assertEqual(FrozenEvaluationEvidence.load(root).estimate(family,4),expected)
                self.assertEqual(files(root),before)
            finally:root.chmod(0o700);(root/'evidence.sqlite').chmod(0o600);(root/'plan.json').chmod(0o600)

    def test_shared_gate_read_lock_blocks_collection_for_the_entire_read(self):
        with tempfile.TemporaryDirectory() as directory:
            run,family=self.create(directory)
            with evaluation_read_lock(directory):
                with self.assertRaises(BlockingIOError):run.run(8,time_cap_seconds=30)
                self.assertEqual(FrozenEvaluationEvidence.load(directory).largest_completed_look,4)
            with run._lock():
                with self.assertRaises(BlockingIOError):
                    with evaluation_read_lock(directory):pass
            run.run(8,time_cap_seconds=30)
            self.assertEqual(FrozenEvaluationEvidence.load(directory).largest_completed_look,8)

    def test_appended_later_look_and_in_memory_changes_invalidate_an_old_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            run,family=self.create(directory);old=FrozenEvaluationEvidence.load(directory)
            run.run(8,time_cap_seconds=30)
            with self.assertRaisesRegex(ValueError,'latest completed look'):old.estimate(family,4)
            latest=FrozenEvaluationEvidence.load(directory)
            self.assertEqual(latest.estimate(family,8),run.estimate(family,8))
            latest.rows[0]['utility']+=.5
            with self.assertRaisesRegex(ValueError,'mutated in memory'):latest.estimate(family,8)

    def test_resealed_wrong_aggregate_or_impossible_terminal_length_is_rejected(self):
        changes=(lambda r:r.update(utility=r['utility']+.25),
                 lambda r:r['subject'][0].update(termination='round_limit',rounds=1),
                 lambda r:r['subject'][0].update(termination='mutual_pass',rounds=1))
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                self.create(directory);alter_unit(directory,change)
                with self.assertRaises(ValueError):FrozenEvaluationEvidence.load(directory)

    def test_actual_episode_mixture_draws_and_matched_incumbent_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            run,family=self.create(directory,mixture=True,incumbent=True)
            evidence=FrozenEvaluationEvidence.load(directory)
            self.assertEqual(evidence.estimate(family,4),run.estimate(family,4))
            alter_unit(directory,lambda row:row['subject'][0].update(subject_component='0'*64))
            with self.assertRaisesRegex(ValueError,'component'):FrozenEvaluationEvidence.load(directory)

    def test_declared_nonempty_root_retains_its_remaining_round_allowance(self):
        with tempfile.TemporaryDirectory() as directory:
            state=resolve(State(Rules(3,2)),0,1).state
            run,family=self.create(directory,openings=FixedStarts((state,),label='actual-prefix-new-episode'))
            evidence=FrozenEvaluationEvidence.load(directory)
            self.assertEqual(evidence.estimate(family,4),run.estimate(family,4))
            self.assertTrue(all(game['rounds']==1 for row in evidence.rows for game in row['subject']))

    def test_missing_database_or_old_pairing_version_cannot_be_reinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create(root);(root/'evidence.sqlite').unlink()
            with self.assertRaisesRegex(ValueError,'missing'):FrozenEvaluationEvidence.load(root)
            self.assertFalse((root/'evidence.sqlite').exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create(root);path=root/'plan.json';data=json.loads(path.read_text())
            data['evaluation_version']='graygo/paired-evaluation/1';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'version'):FrozenEvaluationEvidence.load(root)


if __name__=='__main__':unittest.main()
