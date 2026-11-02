"""Tabular exact-best-response fictitious-play controls for matrix evaluation.

This is a reference learner, not the later NFSP Q/replay/reservoir control.
"""
import time
import numpy as np
from .matrix_training import matrix_payoffs
from .solving import solve_matrix,value_gaps


def fictitious_play(payoffs, *, iterations=50_000, seed=0):
    a=np.asarray(payoffs,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all() or not all(a.shape) or iterations<1:
        raise ValueError("finite matrix and positive iteration count required")
    row=np.ones(a.shape[0]);column=np.ones(a.shape[1])
    row_rng=np.random.default_rng(seed+201);col_rng=np.random.default_rng(seed+202)
    for _ in range(iterations):
        row_values=a@(column/column.sum())
        col_values=(row/row.sum())@a
        # Both responses use pre-update empirical beliefs.
        i=int(row_rng.choice(np.flatnonzero(row_values==row_values.max())))
        j=int(col_rng.choice(np.flatnonzero(col_values==col_values.min())))
        row[i]+=1;column[j]+=1
    p,q=row/row.sum(),column/column.sum()
    solution=solve_matrix(a)
    gaps=value_gaps(a,p,q,solution.value)
    return {"iterations":iterations,"action_records":2*iterations,"seed":seed,
            "row":p.tolist(),"column":q.tolist(),"value":solution.value,
            "gaps":list(gaps),"matrix_certificate":solution.residual_gap}


def matrix_reference_controls():
    started=time.monotonic();results=[]
    for task in ("asymmetric","rps"):
        for seed in (0,1,2):
            result=fictitious_play(matrix_payoffs(task),seed=seed)
            results.append({"task":task,**result})
    return {"control":"tabular_exact_best_response_fictitious_play","passed":all(max(r["gaps"])<=.02 for r in results),
            "threshold":.02,"elapsed_seconds":time.monotonic()-started,"results":results}
