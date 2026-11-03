import ast
from pathlib import Path
import unittest
import graygo
from tools.task_default_specialization import specialize, ast_sha, default_callers

REFERENCE_AST = 'bbd60ad98c8e70e2f9038d229183744488bdfba0ded7d96045ab5997fcdca8dd'


def caller_files(extra=''):
    return {'__init__.py':b'', 'nfsp_training.py':b'from .task_training import live\n',
            'nfsp_collection.py':b'class FrozenPolicies: pass\n',
            'task_training.py':('from .nfsp_collection import FrozenPolicies\n'
                'def live(q,a): return FrozenPolicies(q,a,version=1,average_ready=True)\n'
                'def restore(q,a): return FrozenPolicies(q,a,version=2,average_ready=True)\n'+extra).encode()}


class DefaultSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(Path(graygo.__file__).parent/'nfsp_collection.py').read_text()

    def test_production_default_specializes_to_pinned_original_full_module_ast(self):
        tree,rules=specialize(self.source)
        self.assertEqual(ast_sha(tree),REFERENCE_AST)
        self.assertEqual(rules['inlined_forwards'],4)
        self.assertEqual(rules['false_branches'],4)

    def test_non_none_default_and_changed_delegate_are_rejected(self):
        for before,after in [('forward_pool=None','forward_pool=0'),
                             ('if self.forward_pool is None:return model(**inputs)','if self.forward_pool is None:return self.q(**inputs)')]:
            with self.subTest(after=after),self.assertRaises(ValueError):specialize(self.source.replace(before,after))

    def test_unrelated_probability_and_serialized_state_changes_are_not_erased(self):
        for before,after in [('probability=1.','probability=.5'),("'version':self.version","'version':self.version+1")]:
            changed=self.source.replace(before,after)
            self.assertNotEqual(changed,self.source)
            tree,_=specialize(changed)
            self.assertNotEqual(ast_sha(tree),REFERENCE_AST)

    def test_unknown_metadata_field_and_additional_pool_assignment_are_rejected(self):
        for changed in [self.source.replace("state.get('frozen_execution')","state.get('another_field')"),
                        self.source.replace('self.forward_pool=forward_pool','self.forward_pool=forward_pool\n        self.forward_pool=forward_pool')]:
            with self.assertRaises(ValueError):specialize(changed)

    def test_live_and_restored_constructor_sites_are_both_checked(self):
        self.assertEqual(len(default_callers(caller_files())),2)
        files=caller_files();files['task_training.py']=files['task_training.py'].replace(b'version=1,',b'forward_pool=None,version=1,')
        with self.assertRaises(ValueError):default_callers(files)

    def test_constructor_aliases_pool_mutation_and_dynamic_kwargs_are_rejected(self):
        for extra in ['alias=FrozenPolicies\n','def mutate(x): x.forward_pool=object()\n',
                      'def escaped(q,a,kwargs): return FrozenPolicies(q,a,**kwargs)\n',
                      'from graygo.nfsp_collection import FrozenPolicies as alias\n',
                      "def mutate(x): setattr(x,'forward_pool',None)\n",
                      'def other(x): return x.FrozenPolicies\n',
                      'FrozenPolicies=object\n']:
            with self.subTest(extra=extra),self.assertRaises(ValueError):default_callers(caller_files(extra))
