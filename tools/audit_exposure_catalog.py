"""Read-only structural, paired-rule and native-feature audit of a frozen catalog."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from graygo.curriculum import ExposureCatalog,ExposureStarts,family_id
from graygo.encoding import encode
from graygo.engine import BLACK,WHITE,GRAY,Rules,State,groups,resolve,score
from graygo.evaluation import seed_for
from graygo.manifest import source_digest,write_report
from graygo.rule_state import RuleStateSpec
from graygo.storage import digest


def require(condition,message):
    if not condition:raise ValueError(message)


def audit(catalog_path,native_library,*,samples=512):
    if type(samples) is not int or not 1<=samples<=4096:raise ValueError('bounded catalog audit sample count required')
    started=time.perf_counter();catalog_path=Path(catalog_path)
    catalog=ExposureCatalog.from_dict(json.loads(catalog_path.read_text()))
    inventory={}
    for split in ('train','heldout'):
        boards=getattr(catalog,split)
        inventory[split]={'families':len(boards),'gray_counts':dict(sorted(Counter(map(lambda b:b.count(GRAY),boards)).items())),
                          'family_ids':[family_id(b) for b in boards]}
        for board in boards:
            require(all(group.liberties for group in groups(board,9)),'dead group in catalog')
    require(not set(inventory['train']['family_ids'])&set(inventory['heldout']['family_ids']),'family leakage')
    reports=[];paired={}
    for collision in ('gray','remove'):
        rules=Rules(collision=collision);reference=RuleStateSpec(rules);native=RuleStateSpec(rules,native_library)
        for split,mode in (('train','mixed'),('heldout','gray-only')):
            starts=ExposureStarts(catalog,split,mode);states=[];counts={'empty':0,'gray':0}
            for seed in range(samples):
                root,identity=starts.sample_with_family(seed,rules)
                key=(split,seed)
                if collision=='gray':paired[key]=(root.board,identity)
                else:require(paired[key]==(root.board,identity),'rule arms receive different initial samples')
                require(not root.history and not root.forbidden_black and not root.forbidden_white and root.pass_streak==root.rounds_played==0,'curriculum metadata was not reset')
                if identity is None:require(GRAY not in root.board and mode=='mixed','invalid empty mixture component');counts['empty']+=1
                else:
                    require(identity in inventory[split]['family_ids'] and identity==family_id(root.board),'sample does not belong to its frozen split')
                    counts['gray']+=1
                rng=np.random.default_rng(seed_for(catalog.seed,'catalog-audit-action',split,seed))
                black=int(rng.choice(root.legal_actions(BLACK)));white=int(rng.choice(root.legal_actions(WHITE)))
                states.extend((root,resolve(root,black,white).state))
            for state in states:
                result=score(state.board,9)
                require(result.black_units+result.white_units==324,'fractional score total differs')
            rows=np.stack([np.frombuffer(reference.codec.pack(state,role),dtype=np.uint8) for state in states for role in (BLACK,WHITE)])
            empty_globals=np.empty((len(rows),0),dtype=np.float32);empty_legal=np.empty((len(rows),0),dtype=np.uint8)
            a=reference.unpack(rows,empty_globals,empty_legal);b=native.unpack(rows,empty_globals,empty_legal)
            for reference_features,native_features in zip(a,b):np.testing.assert_array_equal(reference_features,native_features)
            # Independently encode the original public states, bypassing packet unpack.
            for index,(state,role) in enumerate((state,role) for state in states for role in (BLACK,WHITE)):
                observation=encode(state,role)
                for reference_row,expected in zip((part[index] for part in a),(observation.spatial,observation.global_features,observation.legal)):
                    np.testing.assert_array_equal(reference_row,expected)
            reports.append({'rules':collision,'split':split,'mode':mode,'initial_samples':samples,'start_counts':counts,
                'complete_states_checked':len(states),'perspective_observation_rows':len(rows),'packet_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),
                'feature_sha256':hashlib.sha256(b''.join(part.tobytes() for part in b)).hexdigest(),'native_schema':native.to_dict()})
    return {'format_version':1,'verified':True,'catalog_sha256':catalog.sha256,'catalog_file_sha256':digest(catalog_path),
        'source_sha256':source_digest(),'auditor_sha256':digest(Path(__file__)),'catalog_seed':catalog.seed,'inventory':inventory,'samples_per_split_per_rule':samples,
        'checks':reports,'elapsed_seconds':time.perf_counter()-started,'candidate_policy_used':False,'training_started':False,
        'collision_probe_families_certified':0,'scope':'synthetic-root independence/validity, paired initial samples, real one-round transitions, exact score totals and reference/native public features; no scientific performance or full paired-training claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('catalog');parser.add_argument('--native-library',required=True)
    parser.add_argument('--samples',type=int,default=512);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=audit(args.catalog,args.native_library,samples=args.samples);write_report(args.output,result)
    print(json.dumps({k:result[k] for k in ('verified','catalog_sha256','samples_per_split_per_rule','elapsed_seconds','collision_probe_families_certified')},sort_keys=True))
