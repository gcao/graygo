# Positive pool-gain policy verification

All 95 relevant tests pass, including strict zero/negative lower-bound rejection,
unchanged confidence statistics and regression safeguards, old-decision
reproduction, complete-evidence validation and publication without changing the
learner checkpoint.

The real retained 7,340,032-record evaluation reproduces exactly:
- head win/draw score 0.963; lower confidence bound 0.9171216209500461;
- pooled utility gain lower bound 0.003647688102296681;
- all policy-v2 criteria pass; the original v1 pool minimum of 0.03 fails.

The new assessment references the original sealed head/random/tactical evidence.
It does not rewrite the original decision or claim that it passed the old rule.
The owner explicitly approved adoption and publication of this tested snapshot.
