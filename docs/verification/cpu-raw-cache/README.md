# CPU raw-forward cache qualified for bounded 3x3 tasks

The [source-admission receipt](admission/admission.json) qualifies candidate source a714ee0 against a522fe2 for CPU/features gray-pure and gray-mixed tasks. The cache stores immutable raw neural outputs for repeated scalar public inputs. Action masks, probabilities, labels, RNG, learning and serialization retain their original semantics. The table path remains unchanged. Other devices, board sizes and task families fall outside this admission.

Four actual 32,768-record pre/post neural pipelines match the original full partition comparisons exactly: model and target tensors, optimizers, replay/reservoir, metrics, private/global RNG, public archives and episode bytes. All four prefixes pass the unchanged independent bounded auditor. These are diagnostic prefixes, not complete 15M trajectories or new A2 cases.

The original [533-test suite failed](failed-suite/job.json) in two historical fixture tests that incorrectly selected the newly changed source. The fixture correction pins the original delegate source and adds a rejection regression; it leaves the certified verifier and application unchanged. The subsequent [536-test complete suite passed](passing-suite/job.json). Both original logs and costs remain preserved.

The original comparison wrapper loaded a 960-second alarm despite its 780-second allocation. An identity-bound [separate supervisor](comparison/outer-cap-supervisor.json) enforced the declared 780-second limit without editing or restarting either original worker. The original job completed in about 288 seconds. This discrepancy is retained.

Candidate profile wall times were about 110/126 seconds (pure) and 134/148 seconds (mixed), compared with the prior source's 169/185 and 190/204 seconds. Host background load differed; these observations do not establish an isolated universal speedup or independently qualify complete-case runtime prices.

[Retention](retention.json) seals copied metadata and logs. [Binary inventory](raw-binary-inventory.json) seals private checkpoints, episodes and public model payloads at their original permanent raw roots. [Resource settlement](resource-use.json) appends each measured child/process interval once, including both GPU-equipped suite intervals. Control commitment remains 42,236.08177829580345144 of 43,200 CPU process seconds; both active delayed-neural reservations remain. The inventory is partial and identifies unmeasured overhead.

No full case, cross-source resume, control-budget increase, CUDA/native/9x9 admission, pilot or study is authorized by this receipt. Original source-specific plans and independent case audits remain required, with separate runtime-price qualification.
