from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.board_runtime import BoardRuntimeConfig,BoardNFSPRuntime
from graygo.cli import main
from graygo.curriculum import create_catalog,ExposureCatalog,ExposureStarts,canonical_board,family_id
from graygo.engine import BLACK,WHITE,GRAY,Rules,State,groups,neighbors,score,transform
from graygo.manifest import write_report
from graygo.storage import digest


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=create_catalog(8675309,train_families=12,heldout_families=12)

    def test_catalog_is_deterministic_seeded_balanced_and_liberty_valid(self):
        c=self.catalog
        self.assertEqual(ExposureCatalog.from_dict(c.to_dict()),c)
        self.assertEqual(create_catalog(c.seed,train_families=12,heldout_families=12).sha256,c.sha256)
        self.assertNotEqual(create_catalog(c.seed+1,train_families=12,heldout_families=12).sha256,c.sha256)
        totals=[]
        for board in (*c.train,*c.heldout):
            state=State(Rules(),board);gray=board.count(GRAY);totals.append(gray)
            self.assertIn(gray,range(1,7));self.assertEqual(board.count(BLACK),board.count(WHITE))
            self.assertTrue(all(g.liberties for g in groups(board,9)))
            self.assertTrue(all(len(g.stones)<=3 for g in groups(board,9) if g.color==GRAY))
            self.assertEqual(score(board,9).black_units+score(board,9).white_units,324)
            self.assertFalse(state.history);self.assertEqual(state.rounds_played,0)
        self.assertEqual(sorted(totals),sorted(list(range(1,7))*4))

    def test_family_identity_removes_torus_d4_and_color_copies(self):
        for board in self.catalog.train[:3]:
            identity=family_id(board)
            for rotation in range(4):
                for reflect in (False,True):
                    for swap in (False,True):
                        moved=transform(State(Rules(),board),rotation=rotation,reflect=reflect,shift=(7,3),swap=swap)
                        self.assertEqual(family_id(moved.board),identity)
        train={family_id(b) for b in self.catalog.train};heldout={family_id(b) for b in self.catalog.heldout}
        self.assertFalse(train&heldout)
        with self.assertRaisesRegex(ValueError,'equivalent'):ExposureCatalog(0,self.catalog.train,(self.catalog.train[0],))

    def test_catalog_rejects_relabelled_seed_ids_and_invalid_groups(self):
        for key,value in (('seed',self.catalog.seed+1),('max_gray_group',4),('version','unknown')):
            data=deepcopy(self.catalog.to_dict());data[key]=value
            with self.assertRaises(ValueError):ExposureCatalog.from_dict(data)
        data=self.catalog.to_dict();del data['seed']
        with self.assertRaisesRegex(ValueError,'schema'):ExposureCatalog.from_dict(data)
        data=self.catalog.to_dict();data['splits']['heldout']=[{}]
        with self.assertRaisesRegex(ValueError,'schema'):ExposureCatalog.from_dict(data)
        data=deepcopy(self.catalog.to_dict());data['splits']['heldout'][0]['family_id']='0'*64
        with self.assertRaises(ValueError):ExposureCatalog.from_dict(data)
        board=[0]*81;board[0]=GRAY
        for index,point in enumerate(neighbors(9)[0]):board[point]=BLACK if index<2 else WHITE
        with self.assertRaisesRegex(ValueError,'zero-liberty'):ExposureCatalog(0,(canonical_board(board),),self.catalog.heldout)
        board=[0]*81
        for point in (0,1,2,3):board[point]=GRAY
        with self.assertRaisesRegex(ValueError,'groups of at most three'):ExposureCatalog(0,(canonical_board(board),),self.catalog.heldout)
        with self.assertRaisesRegex(RuntimeError,'budget exhausted'):create_catalog(0,train_families=1,heldout_families=1,max_attempts=1)

    def test_matched_rule_samples_have_identical_boards_families_and_fresh_metadata(self):
        starts=ExposureStarts(self.catalog);counts={'gray':0,'empty':0}
        allowed={family_id(board) for board in self.catalog.train}
        for seed in range(256):
            a,fa=starts.sample_with_family(seed,Rules());b,fb=starts.sample_with_family(seed,Rules(collision='remove'))
            self.assertEqual(a.board,b.board);self.assertEqual(fa,fb)
            self.assertEqual(a.rules.collision,'gray');self.assertEqual(b.rules.collision,'remove')
            self.assertEqual(a.rounds_played,0);self.assertEqual(a.pass_streak,0)
            self.assertFalse(a.history);self.assertFalse(a.forbidden_black);self.assertFalse(a.forbidden_white)
            if fa is None:self.assertNotIn(GRAY,a.board);counts['empty']+=1
            else:self.assertIn(fa,allowed);self.assertEqual(fa,family_id(a.board));counts['gray']+=1
        self.assertGreater(counts['gray'],100);self.assertGreater(counts['empty'],100)
        heldout=ExposureStarts(self.catalog,'heldout','gray-only');heldout_ids={family_id(b) for b in self.catalog.heldout}
        self.assertNotEqual(starts.sha256,heldout.sha256)
        for seed in range(32):
            state,identity=heldout.sample_with_family(seed,Rules())
            self.assertIn(identity,heldout_ids);self.assertIn(GRAY,state.board);self.assertNotIn(identity,allowed)
        with self.assertRaises(ValueError):starts.sample(1,Rules(3,200))
        with self.assertRaises(ValueError):starts.sample(1,Rules(9,1))

    def test_actual_collector_resume_keeps_catalog_split_family_counts_and_weights(self):
        torch.set_num_threads(1)
        config=BoardRuntimeConfig(envs=4,seed=17,batch=8,learning_start=8,capacity=64,target_every=16,
            width=8,blocks=1,norm_groups=2,reservoir_open=0,handoff_end=8)
        starts=ExposureStarts(self.catalog)
        full=BoardNFSPRuntime(config,starts=starts);full.run_profile(64,time_cap_seconds=60)
        part=BoardNFSPRuntime(config,starts=starts);part.run_profile(24,time_cap_seconds=60)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=part.save(directory,reserve_bytes=0)
            restored=BoardNFSPRuntime(config,starts=ExposureStarts(ExposureCatalog.from_dict(self.catalog.to_dict())))
            restored.restore(checkpoint);restored.run_profile(64,time_cap_seconds=60)
            self.assertEqual(full.collector.metrics,restored.collector.metrics)
            self.assertEqual(full.collector.start_families,restored.collector.start_families)
            metrics=restored.collector.metrics
            self.assertEqual(sum(metrics['start_counts'].values()),restored.collector.started)
            self.assertEqual(sum(metrics['starting_gray_counts']),restored.collector.started)
            self.assertEqual(sum(metrics['family_starts'].values()),metrics['start_counts']['gray'])
            for key,value in full.backend.q.state_dict().items():torch.testing.assert_close(value,restored.backend.q.state_dict()[key],rtol=0,atol=0)
            for key,value in full.replay.arrays.items():np.testing.assert_array_equal(value,restored.replay.arrays[key])
            wrong=BoardNFSPRuntime(config,starts=ExposureStarts(self.catalog,'heldout'))
            with self.assertRaisesRegex(ValueError,'start distribution differs'):wrong.restore(checkpoint)

    def test_cli_preserves_a_frozen_catalog_and_refuses_changed_generation_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'catalog.json'
            args=['generate-exposure','--catalog',str(path),'--seed','19','--train-families','6','--heldout-families','6']
            def write(path,value):return write_report(path,value,reserve_bytes=0)
            with patch('graygo.cli.write_report',side_effect=write),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),0);before=digest(path)
                self.assertEqual(main(args),0);self.assertEqual(digest(path),before)
                changed=args.copy();changed[changed.index('19')]='20'
                with self.assertRaises(SystemExit) as caught:main(changed)
                self.assertEqual(caught.exception.code,1);self.assertEqual(digest(path),before)
            catalog=ExposureCatalog.from_dict(json.loads(path.read_text()))
            self.assertEqual(catalog.seed,19)

if __name__=='__main__':unittest.main()
