# Champion-loop adoption verification

The final controller/evaluation/checkpoint test group passes 37 tests with one CUDA skip. Tests verify complete learner/optimizer/RNG/buffer/trajectory preservation after rejection, fixture-only local promotion, checkpoint resume, interrupted evaluation, actual independent color-paired matches, historical regression checks and atomic/stale publisher behavior. Synthetic promotion fixtures are confined to test directories and are not model-strength evidence.

The real two-record bootstrap used the full default 9x9 board/model shape and copied old checkpoint 0081 as the frozen incumbent. The average learner was not ready, so no candidate was promoted. Its exact controller source is retained separately because a later release-file fsync hardening changed the controller hash; test and bootstrap bindings are not relabeled.

The arena's historical and loop slots were each health-checked and played three human-versus-model turns with valid responses and the expected checkpoint checksum. Existing 32k NFSP demo remains separate. Model revisions are held fixed for each game.

No new long training or formal A2/promotion grant is claimed. The public CLI remains a bounded diagnostic; the complete project objective and paused goal are unchanged.
