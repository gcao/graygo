# Stationary forced-opponent diagnostic

Source is the original frozen `9fefeee` package, SHA `ba5646f14d4aad47dbd9e7db33b9fad2c1cc1115df0e989c55522218054a5f6e`. The tracked `tools/diagnose_forced_opponent.py` exercises the actual old joint collector, replay decoding, expected one-step targets and Q loss. Original artifacts remain on `ai` at `/home/gcao/proj/graygo/runs/forced-opponent-diagnostic-9fefeee`.

This is a controlled oracle-initialized loss fixture, not a learned-convergence or new full-budget selection trial. It uses 44 fixture own-side records across both controller orientations and two demonstration gradient steps. All results are source/driver bound; copied files match the original bytes. The live R4 comparison is unchanged.
