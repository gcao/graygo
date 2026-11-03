# Persistent learner and champion releases

The controller in tools/champion_loop.py adopts the old generation/train/gate/export cycle while preserving NFSP's persistent Q, target and average networks, Adam state, replay, averaging reservoir, record clocks, active frozen episode actors and private RNGs. Candidate rejection and promotion affect only frozen release metadata; they never reload the learner from a champion or flush its buffers.

A fixed list of record endpoints is declared before training. The existing BoardExecutor owns locking, cumulative time, safe checkpoints, opening barriers and storage reserves. At each endpoint the controller exports an immutable average-policy candidate. Uninitialized or identical average weights are not evaluated as new improvements.

The gate in tools/champion_gate.py uses the existing independent-seat, paired-color evaluator. It requires a lower confidence bound on head-to-head win/draw score above 50%, aggregate gain against a frozen pool, and no unacceptable historical/baseline regressions. All-draw and sub-50% results fail. Alpha is divided across the fixed rounds, endpoints and planned looks; it is a finite-loop guarantee, not a lifetime guarantee across arbitrary new runs.

The pool contains random, tactical and retained historical champions. Incumbent and historical identities are frozen before the training block. Evaluation inputs contain only the same immutable pre-turn state; each actor has private action draws. The deployed raw policy is evaluated, not a shared joint-MCTS controller. Existing completed later looks cannot be hidden, and interrupted evaluations resume without replaying training.

Loop state and champion publication are checksummed atomic JSON. Policy payloads are immutable and verified against their archived weights. The champion pointer is published only after the loop state commits, so recovery can repair a lagging pointer. A single-publisher lease and contract identity prevent a stale resumed run from reverting another loop's champion. Private learner checkpoints remain owned by BoardExecutor.

Diagnostic mode may publish a clearly experimental local champion after numerical criteria pass. Formal mode additionally requires verified project entry and the existing calibrated protected-promotion authorizer; a numerical gate result alone cannot issue an A2, training or formal promotion grant. The public CLI remains bounded to 65,536 records and one cumulative hour. The full project scope and original learning/scientific gates remain unchanged.

Example bounded diagnostic command (not launched automatically):

~~~sh
PYTHONPATH=src:. .venv/bin/python -m tools.champion_loop --diagnostic --run-dir runs/league-diagnostic --records 65536 --block-records 16384 --time-cap-seconds 3600 --gate-seconds 600 --gate-looks 500 2000 --device cuda --reservoir-open 0 --handoff-end 1000 --incumbent-release /home/gcao/proj/graygo-arena/.runtime/models/champion.json --publish-manifest /home/gcao/proj/graygo-arena/.runtime/models/champion.json
~~~

The early opening in this example is an explicit diagnostic variation. It is not the primary 5M/7M protocol. A new fixed run can inherit the current release as its evaluation incumbent without inheriting or resetting that model's training state. Resume uses the original run directory, configuration, source and round plan.

A two-record bootstrap smoke test initialized the arena pointer to a frozen copy of graygo-old champion_iter_0081.pt. No Q/supervised optimizer update or new promotion was claimed. Its exact pre-fsync-hardening controller source is retained in runs/champion-loop-bootstrap-v1/implementation. Longer training has not been started.

Regression tests compare complete real learner state/buffers/trajectories across rejection, promotion fixtures, round resume and interrupted gates; they also check confidence rejection, historical regression, atomic publication and stale-publisher rejection.
