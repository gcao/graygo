"""Measure compact-buffer storage and IO using legal game data, not training."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
from .buffers import ObservationSpec,ReplayBuffer,ReservoirBuffer,spec_metadata
from .engine import BLACK,WHITE,State,resolve
from .encoding import encode
from .storage import CheckpointStore


def profile_buffers(rules,root,*,capacity=500000,records=10000,iterations=50,batch=512,seed=42,reserve_bytes=20*1024**3,storage="features",native_library=None):
    if records<2 or records%2 or iterations<1 or batch<1:
        raise ValueError("positive dimensions and even fresh-record count required")
    if native_library is not None and storage!='rule-state':raise ValueError('native decoding requires rule-state storage')
    if storage not in ('features','rule-state'):raise ValueError('unknown observation storage')
    if storage=='rule-state':
        from .rule_state import RuleStateSpec,RuleObservation
        spec=RuleStateSpec(rules,native_library);observation=RuleObservation
    else:spec=ObservationSpec.board(rules.size);observation=encode
    replay=ReplayBuffer(capacity,spec);reservoir=ReservoirBuffer(capacity,spec)
    store=CheckpointStore(root,reserve_bytes=reserve_bytes)
    projected=(replay.allocated_bytes+reservoir.allocated_bytes)*min(records,capacity)//capacity+131072
    store.ensure_space(projected)
    row_rng=random.Random(seed);col_rng=random.Random(seed+1);rng=np.random.default_rng(seed+2)
    state=State(rules);episode=0;record=0;started=time.perf_counter()
    while record<records:
        actions=(row_rng.choice(state.legal_actions(BLACK)),col_rng.choice(state.legal_actions(WHITE)))
        result=resolve(state,*actions)
        for slot,player in enumerate((BLACK,WHITE)):
            obs,nxt=observation(state,player),observation(result.state,player)
            replay.add(obs,actions[slot],result.reward_black*(1 if player==BLACK else -1),nxt,result.state.done,
                       behavior_prob=1/len(state.legal_actions(player)),stream=episode*2+slot,step=state.rounds_played)
            # Artificial all-legal tie labels benchmark storage only, not Q quality.
            reservoir.offer(obs,obs.legal,record=record,policy_version=0,rng=rng,best_response_mode=True)
            record+=1
        state=result.state
        if state.done:episode+=1;state=State(rules)
    collection=time.perf_counter()-started
    started=time.perf_counter()
    for _ in range(iterations):replay.sample(batch,rng)
    replay_seconds=time.perf_counter()-started
    started=time.perf_counter()
    for _ in range(iterations):reservoir.sample(batch,rng)
    reservoir_seconds=time.perf_counter()-started
    def write(path):
        replay.save(path/'replay.npz');reservoir.save(path/'reservoir.npz')
        (path/'sampler.json').write_text(json.dumps(rng.bit_generator.state))
    started=time.perf_counter()
    bundle=store.save(records,write,metadata={"schema":spec_metadata(spec),"synthetic_labels":True},estimated_peak_bytes=projected)
    checkpoint_seconds=time.perf_counter()-started
    started=time.perf_counter()
    loaded=ReplayBuffer.load(bundle/'replay.npz');loaded_reservoir=ReservoirBuffer.load(bundle/'reservoir.npz')
    restore_seconds=time.perf_counter()-started
    for a,b in ((replay.sample(16,np.random.default_rng(9)),loaded.sample(16,np.random.default_rng(9))),
                (reservoir.sample(16,np.random.default_rng(9)),loaded_reservoir.sample(16,np.random.default_rng(9)))):
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    return {"storage":storage,"schema":spec_metadata(spec),"capacity_per_buffer":capacity,"fresh_records":records,"filled_rows":replay.size,
            "partial_capacity_fill":records<capacity,"completed_games":episode,
            "replay_logical_allocation_bytes":replay.allocated_bytes,"reservoir_logical_allocation_bytes":reservoir.allocated_bytes,
            "collection_seconds":collection,"sample_batch":batch,"sample_iterations":iterations,
            "replay_decode_seconds":replay_seconds,"reservoir_sample_seconds":reservoir_seconds,
            "checkpoint_seconds":checkpoint_seconds,"restore_validation_seconds":restore_seconds,"checkpoint_payload_bytes":sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),
            "roundtrip_verified":True,"label_generation":"synthetic all-legal ties, all records eligible for storage benchmark",
            "scope":"buffer IO only; neither a complete learner profile nor an isolated campaign runtime estimate"}
