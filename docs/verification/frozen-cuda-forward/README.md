# Opt-in shared frozen CUDA forwards

The actor/evaluation timing evidence identifies many small immutable-network launches as a remaining cost. `FrozenForwardPool` reuses a captured official-model forward across frozen weight versions: it copies the selected immutable parameter vector and public inputs into fixed buffers, replays the graph, and clones the output. It captures no sampler, legality logic, optimizer, target update or fresh-record operation.

## Capture and numerical contract

PyTorch graphs require static shapes/control flow, stable input/output addresses and side-stream warm-up; CPU/GPU synchronization checks stay outside capture ([PyTorch 2.11 CUDA graph guidance](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)). This executor keeps three warm-up forwards, long-lived buffers and a cloned result. Capture verifies global CPU/CUDA RNG states are unchanged. Different frozen models share graph code/storage while each replay copies its selected full parameter vector. Parameter views are aligned to 256 bytes.

Only the exact full 64-channel/eight-block/GN8 QNet/PolicyNet on 9x9 boards, FP32 and batch sizes 1..32 enter capture. The two classes and 32 batches bound template inventory to 64 per worker. Other shapes use the recorded eager fallback. Weight packs are weakly owned by immutable models; a version/storage change rejects stale reuse. Explicit checkpoint loading invalidates/rebuilds weight packs and reconstructs caches. CUDA graphs themselves never enter checkpoints or public weights.

A pool has one owner thread, stream and deterministic runtime context. Changed context is rejected. The supported runtime is the existing single-worker CUDA execution path; no concurrent noncaptured CUDA work is introduced during capture. Normal training and backpropagation remain on the shared eager learner path. Fixed evaluation batch shape and public-observation row placement remain unchanged, so single/probe/grouped calls retain the same selected numerical function.

## Configuration and evidence bindings

`BoardRuntimeConfig.cuda_graphs` defaults false. The actor snapshot records the selected execution descriptor and validates it on restore. Its active models share one worker pool across versions. `FrozenNeuralPolicy(..., cuda_graphs=True)` freezes the descriptor into its identity; paired/scientific/probe resume and public-archive checks bind that choice. Cohort registration fixes it before training and read-only gate verification reconstructs the declared identity on CPU without allocating CUDA or sampling games. Default snapshot/policy metadata retains the previous eager schema.

The opt-in CLI flag is `--cuda-graphs` on `profile-board`, `run-board-diagnostic`, frozen `evaluate`, `science-evaluate`, `probe-evaluate` and `cohort-register`. Probe audit recovers it from saved policy identities. CPU/nondeterministic board configurations reject the flag. Core-only baseline evaluation keeps its NumPy-only path. None of these commands grants A2, resource or production scope.

## Current verification

The bounded full-network prototype compares three different weight versions per class at batches 1/2/32: all 72 output comparisons are bit identical. Matching forward timings include weight/input copies and output cloning. Median raw forward improvements on this hardware are about 3.3–5.0x; cold capture/warm-up is separately retained. [Code](prototype.py) and [raw results](prototype-report.json) use synthetic full-sized networks/inputs and supply no campaign price.

Nine executor checks pass, covering template sharing across versions, unchanged global RNG, non-aliased outputs, stale-weight rejection, explicit invalidation, runtime changes, fallback accounting, sixteen private q/average snapshot versions, exact probabilities/greedy labels/private RNG, fixed-policy marginals and full-model checkpoint/optimizer/buffer continuation. A further cohort case binds the graph selection and verifies its score receipt entirely on CPU. The complete suite passes **430 tests in 302.171 seconds** against unchanged package `a522fe224ccf6db1c528d3fea1c896869038b29e48dbda9e00174ca8116a24df`; original driver/test PIDs **2635473 / 2635497** are terminal. The [full log](suite/tests.txt) and [job binding](suite/job.json) are retained. All thirteen isolated cohort checks also pass with deterministic setup before their first CUDA load. A separately [declared real CLI runtime/evaluation diagnostic](../../experiments/frozen-cuda-runtime-diagnostic-allocation.json) is the next verification step and grants no formal learning.

An isolated cohort test initially loaded CUDA before setting deterministic CUBLAS configuration and failed its later setup check. Its setup now configures the selected runtime before the first CUDA load; the original full-suite source is unchanged. The repeat verifies the same gates and does not weaken runtime checks.

## Pricing remains separate

Finite forward timing does not replace the measured H values. Complete pricing must separate steady replay from cold graph capture and per-snapshot parameter packing, include checkpoint reconstruction and model/evaluation process startup, and count their bounded work explicitly. Existing source-bound cost overruns remain authoritative until an allocated complete selected-source measurement and independent audit review the new path. The original R4 neural case remains on its original frozen CPU source/process and full limits.
