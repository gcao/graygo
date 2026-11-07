"""Versioned average-policy selection; common pure-Python deployment math."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

VERSION='graygo/average-policy-selection/1'


def implementation_sha():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def selection_spec(temperature=0.,placement_only=True,tie_rtol=1e-6):
    value={'version':VERSION,'temperature':temperature,'placement_only':placement_only,
           'tie_rtol':tie_rtol,'implementation_sha256':implementation_sha()}
    validate_selection(value)
    return value


def validate_selection(value):
    if set(value)!= {'version','temperature','placement_only','tie_rtol','implementation_sha256'}:
        raise ValueError('unknown policy-selection fields')
    if value['version']!=VERSION or value['implementation_sha256']!=implementation_sha():
        raise ValueError('policy-selection implementation differs')
    t,r=value['temperature'],value['tie_rtol']
    if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=1:
        raise ValueError('finite temperature in [0,1] required')
    if type(value['placement_only']) is not bool or isinstance(r,bool) or not isinstance(r,(int,float)) or not math.isfinite(r) or not 0<=r<=1e-3:
        raise ValueError('invalid policy-selection parameters')


def select_weights(weights,selection):
    """PASS is the last action; zero weights remain impossible actions."""
    validate_selection(selection)
    weights=[float(value) for value in weights]
    if len(weights)<2 or any(not math.isfinite(value) or value<0 for value in weights):
        raise ValueError('finite nonnegative action weights required')
    total=math.fsum(weights)
    if total<=0:raise ValueError('nonempty action distribution required')
    weights=[value/total for value in weights]
    placement_only=selection['placement_only'];domain=weights[:-1] if placement_only else weights
    maximum=max(domain)
    if maximum==0:return weights
    temperature=selection['temperature']
    if temperature==0:
        chosen=[1. if value>0 and value>=maximum*(1-selection['tie_rtol']) else 0. for value in domain]
    else:chosen=[(value/maximum)**(1/temperature) for value in domain]
    denominator=math.fsum(chosen)
    mass=1-weights[-1] if placement_only else 1.
    result=[mass*value/denominator for value in chosen]
    if placement_only:result.append(weights[-1])
    return result


class SelectedAveragePolicy:
    def __init__(self,base,selection):
        if getattr(base,'representation',None)!='average':raise ValueError('only average policies support this selection')
        validate_selection(selection)
        self.base=base;self.selection=deepcopy(selection);self.label=base.label
        self.identity=deepcopy(base.identity)
        self.identity.update(base_policy_sha256=base.sha256,selection=deepcopy(selection))
        self.sha256=canonical_sha(self.identity)
    def assert_frozen(self):
        self.base.assert_frozen();validate_selection(self.selection)
        expected=deepcopy(self.base.identity);expected.update(base_policy_sha256=self.base.sha256,selection=self.selection)
        if self.identity!=expected or canonical_sha(self.identity)!=self.sha256:
            raise ValueError('selected policy changed')
    def _action_weights(self,state,player):
        import numpy as np
        return np.asarray(select_weights(self.base._action_weights(state,player),self.selection),dtype=float)
    def initial_distribution(self,state,player,*,fresh_episode=False):return self._action_weights(state,player)
    def begin_episode(self,seed):return _Actor(self,seed)


class _Actor:
    def __init__(self,policy,seed):
        import numpy as np
        self.policy=policy;self.rng=np.random.default_rng(seed);self.component=None
    def action(self,state,player):
        from graygo.sampling import draw_actions
        actions,_=draw_actions(self.policy._action_weights(state,player)[None],self.rng)
        return int(actions[0])


def load_selection_module(path,expected_sha):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:raise ValueError('frozen selection sidecar changed')
    spec=importlib.util.spec_from_file_location('graygo_policy_selection_'+expected_sha[:16],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def apply_selection(base,selection,bundle=None):
    if selection is None:return base
    if bundle is not None:
        source=Path(bundle).with_name('policy_selection.py')
        module=load_selection_module(source,selection['implementation_sha256'])
        return module.SelectedAveragePolicy(base,selection)
    return SelectedAveragePolicy(base,selection)
