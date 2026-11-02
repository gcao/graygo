"""Optional differential audit against the reviewer gg.cpp ABI.

The external prototype is evidence, not a production dependency or rule oracle.
Hand-authored rule fixtures and symmetry tests remain independent of this audit.
"""
import ctypes
from pathlib import Path
import random
import numpy as np
from .engine import BLACK,WHITE,State,resolve
from .encoding import encode
from .storage import digest


def compare_native(library, *, rounds=3000, seed=812):
    lib=ctypes.CDLL(str(Path(library).resolve()))
    pointer=ctypes.c_void_p
    lib.gg_new.restype=pointer
    lib.gg_new.argtypes=[ctypes.c_int,ctypes.c_uint64]
    lib.gg_encode.argtypes=[pointer,pointer,pointer]
    lib.gg_step.argtypes=[pointer,pointer,pointer,pointer,pointer,pointer]
    env=lib.gg_new(1,seed)
    if not env:raise RuntimeError("native allocation failed")
    spatial=np.zeros((1,2,30,81),np.uint8)
    globals=np.zeros((1,2,14),np.float32)
    reward=np.zeros(1,np.float32);done=np.zeros(1,np.uint8);length=np.zeros(1,np.int32)
    def ptr(a):return a.ctypes.data_as(pointer)
    rng=random.Random(seed); state=State();games=0
    for step in range(rounds):
        lib.gg_encode(env,ptr(spatial),ptr(globals))
        for slot,player in enumerate((BLACK,WHITE)):
            obs=encode(state,player)
            np.testing.assert_array_equal(spatial[0,slot].reshape(30,9,9),obs.spatial,err_msg=f"step {step} spatial")
            np.testing.assert_array_equal(globals[0,slot],obs.global_features,err_msg=f"step {step} globals")
        b=np.array([rng.choice(state.legal_actions(BLACK))],np.int32)
        w=np.array([rng.choice(state.legal_actions(WHITE))],np.int32)
        result=resolve(state,int(b[0]),int(w[0]))
        lib.gg_step(env,ptr(b),ptr(w),ptr(reward),ptr(done),ptr(length))
        if bool(done[0]) != result.state.done or int(reward[0]) != result.reward_black:
            raise AssertionError(f"native terminal/reward mismatch at round {step}")
        if result.state.done:
            if int(length[0]) != result.state.rounds_played:raise AssertionError("native length mismatch")
            games+=1;state=State()
        else:state=result.state
    # The reviewer ABI has no destructor; one process-scoped allocation is used.
    return {"rounds":rounds,"completed_games":games,"seed":seed,
            "external_library_sha256":digest(Path(library)),"mismatches":0}
