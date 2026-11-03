"""The raw-cache checker changes only its declared diagnostic recipe."""
import ast
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import task_pipeline_parity,cpu_cache_pipeline


class CPUCachePipelineContractTests(unittest.TestCase):
    def test_complete_comparison_body_and_all_exclusions_are_preserved(self):
        expected=inspect.getsource(task_pipeline_parity.compare_case)
        expected=expected.replace("config['task'] == 'delayed'","config['task'] in ('gray-pure','gray-mixed') and config['learner']=='neural'")
        expected=expected.replace("config['records'] == 65536","config['records'] == 32768").replace("config['eval_every'] == 32768","config['eval_every'] == 16384")
        expected=expected.replace("result['records'] == 65536","result['records'] == 32768").replace('[32768, 65536]','[16384, 32768]').replace('[65536]','[32768]').replace("'records_per_source': 65536","'records_per_source': 32768")
        actual=inspect.getsource(cpu_cache_pipeline.compare_case)
        self.assertEqual(ast.dump(ast.parse(actual),include_attributes=False),ast.dump(ast.parse(expected),include_attributes=False))

    def test_all_four_original_source_roots_are_required_and_passed_unchanged(self):
        cases=[{'phase':p,'config':{'task':t,'learner':'neural'}} for t in ('gray-pure','gray-mixed') for p in ('pre','post')]
        allocation={'cases':cases,'reference_source_sha256':'a'*64,'candidate_source_sha256':'b'*64}
        with patch.object(cpu_cache_pipeline,'compare_case',return_value={'verified':True}) as compare:
            result=cpu_cache_pipeline.compare_profiles('/old','/new',allocation);self.assertEqual(compare.call_count,4)
            first=compare.call_args_list[0];self.assertEqual(first.args,(Path('/old/gray-pure/neural/neural-R4-pre'),Path('/new/gray-pure/neural-R4-pre')));self.assertEqual(first.kwargs['reference_source'],'a'*64);self.assertEqual(first.kwargs['candidate_source'],'b'*64)
            self.assertEqual(result['aggregate_records_compared'],262144);self.assertFalse(result['source_compatibility_qualified']);self.assertFalse(result['execution_authorized'])
        with patch.object(cpu_cache_pipeline,'compare_case',return_value={}):
            with self.assertRaises(ValueError):cpu_cache_pipeline.compare_profiles('/old','/new',allocation|{'cases':cases[:-1]})
            with self.assertRaises(ValueError):cpu_cache_pipeline.compare_profiles('/old','/new',allocation|{'cases':cases+[cases[0]]})


if __name__=='__main__':unittest.main()
