"""Source admission for equivalent CPU matrix collection implementations.

Preserves old case/source identities. This checks implementation compatibility,
not a fresh learning gate, a runtime price or authority to launch experiments.
"""
import argparse
import ast
from copy import deepcopy
from importlib import metadata
import json
from pathlib import Path
import time
from tools.task_actor_boundary import file_sha,require
from tools.task_source_diff import inventory,normalized
from tools.profile_terminal_batch import compare_pair

VERSION='graygo/terminal-batch-source-admission/1'
WRAPPER='''def collect(self,count):
    if self.collection_implementation=='scalar-reference':return self.collect_reference(count)
    return self.collect_batched(count)
'''
SCALAR_APPEND='''self.replay.add(obs,int(actions[index]),float(rewards[index]),self.terminal,True,behavior_prob=float(probabilities[index]),
    stream=2*(self.episodes+index//2)+role,step=0,opponent_version=opponent_version,policy_version=version,
    mode=int(choice.best_response_mode))'''
BATCH_APPEND='''versions=np.full(count,2*generation,dtype=np.uint64)+avg_rows.cpu().numpy().astype(np.uint64)
self.replay.add_terminal_batch([self.observations[i%2] for i in range(count)],actions,rewards,[self.terminal]*count,
    behavior_prob=probabilities,stream=np.arange(2*self.episodes,2*self.episodes+count,dtype=np.uint64),
    opponent_version=versions.reshape(-1,2)[:,::-1].reshape(-1),policy_version=versions,
    mode=np.fromiter((int(c.best_response_mode) for c in choices),dtype=np.uint8,count=count))'''


def same(a,b):return normalized(a)==normalized(b)


def prove_storage_and_collector(old_files,new_files):
    require(set(old_files)==set(new_files),'package file inventory changed outside the admitted implementation')
    changed={name for name in old_files if old_files[name]!=new_files[name]}
    require(changed=={'buffers.py','nfsp_training.py'},'unreviewed source changes outside terminal collection')
    buffers=ast.parse(new_files['buffers.py']);removed=[];body=[]
    for node in buffers.body:
        if isinstance(node,ast.FunctionDef) and node.name=='_batch_values':removed.append('_batch_values')
        else:body.append(node)
    buffers.body=body
    for cls in (n for n in buffers.body if isinstance(n,ast.ClassDef)):
        wanted={'ObservationSpec':'pack_batch','ReplayBuffer':'add_terminal_batch'}.get(cls.name)
        if wanted:
            nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==wanted];require(len(nodes)==1,'missing expected storage extension')
            cls.body.remove(nodes[0]);removed.append(cls.name+'.'+wanted)
    require(len(removed)==3 and set(removed)=={'_batch_values','ObservationSpec.pack_batch','ReplayBuffer.add_terminal_batch'}
            and same(buffers,ast.parse(old_files['buffers.py'])),'scalar replay or storage format changed')
    old_tree=ast.parse(old_files['nfsp_training.py']);new_tree=ast.parse(new_files['nfsp_training.py'])
    old_cls=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='NFSPMatrixTrainer')
    cls=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='NFSPMatrixTrainer')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    scalar=next(n for n in old_cls.body if isinstance(n,ast.FunctionDef) and n.name=='collect')
    reference=deepcopy(methods['collect_reference']);reference.name='collect';require(same(reference,scalar),'literal reference collector changed')
    require(same(methods['collect'],ast.parse(WRAPPER).body[0]),'collector dispatch differs from the admitted modes')
    expected=deepcopy(scalar);expected.name='collect_batched';found=0
    for position,node in enumerate(expected.body):
        if (isinstance(node,ast.For) and isinstance(node.target,ast.Tuple)
                and [getattr(n,'id',None) for n in node.target.elts]==['index','choice']
                and same(node.iter,ast.parse('enumerate(choices)',mode='eval').body)):
            matches=[item for item in node.body if same(item,ast.parse(SCALAR_APPEND).body[0])];require(len(matches)==1,'scalar append call changed')
            node.body.remove(matches[0]);expected.body[position:position]=ast.parse(BATCH_APPEND).body;found+=1;break
    require(found==1 and same(expected,methods['collect_batched']),'bulk collector changes more than the checked terminal write substitution')
    cls.body.remove(methods['collect']);cls.body.remove(methods['collect_batched']);methods['collect_reference'].name='collect'
    init=methods['__init__'];i=next(i for i,arg in enumerate(init.args.kwonlyargs) if arg.arg=='collection')
    require(same(init.args.kw_defaults[i],ast.Constant('terminal-batch')),'default collector mode changed');init.args.kwonlyargs.pop(i);init.args.kw_defaults.pop(i)
    prefix=ast.parse("if collection not in ('terminal-batch','scalar-reference'):raise ValueError('unknown matrix collection implementation')\nself.collection_implementation=collection").body
    require(all(same(a,b) for a,b in zip(init.body[:2],prefix)),'constructor mode guard changed');del init.body[:2]
    guards=[ast.parse("if self.manifest.get('collection_implementation')!=collection:raise ValueError('matrix collection implementation differs on resume')").body[0],
            ast.parse("if state.get('collection_implementation')!=self.collection_implementation:raise ValueError('checkpoint collection implementation differs')").body[0]]
    counts={'guards':0,'metadata':0}
    class Strip(ast.NodeTransformer):
        def visit_If(self,node):
            if any(same(node,g) for g in guards):counts['guards']+=1;return None
            return self.generic_visit(node)
        def visit_Dict(self,node):
            for i in reversed(range(len(node.keys))):
                if isinstance(node.keys[i],ast.Constant) and node.keys[i].value=='collection_implementation':
                    require(ast.unparse(node.values[i]) in ('collection','self.collection_implementation'),'unexpected execution metadata value');node.keys.pop(i);node.values.pop(i);counts['metadata']+=1
            return self.generic_visit(node)
    new_tree=Strip().visit(new_tree)
    require(counts=={'guards':2,'metadata':3} and same(old_tree,new_tree),'matrix learning, configuration or state changed beyond explicit execution metadata')
    return {'scalar_storage_and_format_preserved':True,'literal_reference_collector_preserved':True,'matrix_other_semantics_preserved':True,
            'bulk_substitution':'same chronological terminal observations/actions/rewards/versions/modes; reservoir callbacks remain ordered and replay sampling/learning occurs only after collection',
            'ring_semantics':'each final slot/ordinal and total/live eligibility count equals scalar insertion; active tails and specialized schemas use scalar fallback'}


def check_matrix_case(manifest,proof):
    from graygo.nfsp_training import NFSPMatrixConfig
    require(proof['version']==VERSION and proof['status']=='qualified_cpu_matrix_terminal_batch','qualified matrix implementation proof required')
    require(manifest['source_sha256']==proof['candidate_source_sha256'] and manifest.get('collection_implementation')=='terminal-batch','source or collector mode not admitted')
    config=NFSPMatrixConfig(**manifest['config']);require(config.task in ('asymmetric','rps') and config.device=='cpu' and type(config.seed) is int
        and config.seed in (0,1,2) and config.formal_protocol(),'matrix case outside unchanged primary CPU recipe')
    require(manifest['dependencies']==proof['dependencies'],'matrix numerical packages differ')
    return True


def qualify(reference,candidate,comparison,audits,suite,*,expected_reference,expected_candidate):
    started=time.monotonic();old_root,old,old_sha=inventory(reference);new_root,new,new_sha=inventory(candidate)
    require(old_sha==expected_reference and new_sha==expected_candidate,'pinned implementation source differs')
    code=prove_storage_and_collector(old,new);comparison=Path(comparison).resolve();audits=Path(audits).resolve();suite=Path(suite).resolve();inputs={}
    def sealed(path):inputs[str(path)]=file_sha(path);return json.loads(Path(path).read_text())
    job=sealed(comparison/'job.json');report=sealed(comparison/'measurement/report.json');audit_job=sealed(audits/'job.json');test_job=sealed(suite/'job.json')
    require(job['status']=='complete_exact_pipeline_pairs' and job['source_sha256']==new_sha and file_sha(comparison/'measurement/report.json')==job['report_sha256'],'original exact pipeline run differs')
    require(report['status']=='complete_exact_pipeline_pairs' and report['source_unchanged'] and len(report['pairs'])==12 and report['diagnostic_records']==2124288,'incomplete exact workload')
    for label,stored in report['pairs'].items():
        actual=compare_pair(comparison/'measurement'/label/'scalar-reference',comparison/'measurement'/label/'terminal-batch')
        require(actual['equal_partitions']==stored['equal_partitions'] and actual['input_files_sha256']==stored['input_files_sha256'],'fresh full pipeline comparison differs')
        for root,entries in actual['input_files_sha256'].items():
            for name,digest in entries.items():inputs[str(Path(root)/name)]=digest
    require(audit_job['status']=='verified_all_24_diagnostics' and audit_job['original_inputs_unchanged'] and len(audit_job['commands'])==24,'independent diagnostic audits incomplete')
    require(audit_job['auditor_sha256']==file_sha(Path(__file__).with_name('audit_nfsp_matrix.py')),'independent auditor implementation differs')
    for row in audit_job['commands']:
        require(row['returncode']==0,'independent audit failed');path=audits/(row['case'].replace('/','-')+'.json');proof=sealed(path)
        require(file_sha(path)==row['result_sha256'] and proof['verified'] and proof['source_sha256']==new_sha and not proof['formal_protocol'],'independent diagnostic binding differs')
    require(test_job['status']=='passed' and test_job['returncode']==0 and test_job['source_sha256']==new_sha and test_job['source_unchanged'],'complete package validation missing')
    inputs[str(suite/'tests.txt')]=file_sha(suite/'tests.txt');require('Ran 493 tests' in (suite/'tests.txt').read_text() and (suite/'tests.txt').read_text().rstrip().endswith('OK'),'full test log differs')
    require(inventory(old_root)[1]==old and inventory(new_root)[1]==new and all(file_sha(p)==digest for p,digest in inputs.items()),'source/evidence changed during admission')
    return {'version':VERSION,'status':'qualified_cpu_matrix_terminal_batch','reference_source_sha256':old_sha,'candidate_source_sha256':new_sha,
        'evidence':{'reference':str(old_root),'candidate':str(new_root),'comparison':str(comparison),'audits':str(audits),'suite':str(suite)},
        'families':['asymmetric','rps'],'seeds':[0,1,2],'learners':['neural','tabular'],'device':'cpu','collection_implementation':'terminal-batch',
        'dependencies':{name:metadata.version(name) for name in ('numpy','scipy','torch')},'static_proof':code,'exact_pipeline_pairs':12,'independent_diagnostic_audits':24,
        'input_files_sha256':inputs,'verifier_sha256':file_sha(__file__),'qualification_seconds':time.monotonic()-started,
        'original_source_identities_preserved':True,'cross_source_resume_allowed':False,'a2_complete':False,'execution_authorized':False,
        'scope':'Same-protocol primary CPU matrix implementation equivalence and mode admission only; retain original source-specific plans/audits. No full-run price, new case, missing seed or project-entry grant.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','candidate','expected-reference','expected-candidate','comparison','audits','suite','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output).resolve();require(not out.exists(),'preserve existing admission receipt')
    result=qualify(a.reference,a.candidate,a.comparison,a.audits,a.suite,expected_reference=a.expected_reference,expected_candidate=a.expected_candidate)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({key:result[key] for key in ('status','exact_pipeline_pairs','qualification_seconds','a2_complete')},indent=2))
