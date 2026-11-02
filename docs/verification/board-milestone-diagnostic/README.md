# Bounded board milestone diagnostic evidence

Measured clean revision: `546d2e27095708ff086b90af6ba3af8ba46a9d04`.
Package source SHA: `18bc62c487c4d90dcad1814b15705a89c3ca743ef8c717f5ffa29119cbad15f6`.
Full retained run: `/home/gcao/proj/graygo/runs/board-journal-R4-diagnostic-546d2e2` on `ai`.

The frozen plan, report, independent audit, run manifests, immutable opening decisions and all public/final bundle manifests are copied byte-for-byte. Heavy models and buffers remain at the retained run path; the JSON copies alone are not restorable model bundles. The audit records the auditor source SHA, final seals, model archive clocks, exact full-state comparison and label accounting.

Run the artifact audit with the measured package source and dependencies on `PYTHONPATH`:

```text
.venv/bin/python tools/audit_board_journal.py runs/board-journal-R4-diagnostic-546d2e2 --output /tmp/graygo-board-journal-audit.json
```

Two full-model CUDA trajectories each reach 16,384 fresh records. They use a diagnostic opening at 8,192 and hand-off at 12,288, reference Python decoding, Q reuse four, batch 512 and 500k-capacity buffers. The second trajectory recovers an opening decision written after its pending checkpoint. Exact runtime/optimizer/collector/RNG state and all allocated buffer arrays match the uninterrupted trajectory. No labels precede opening; 903 eligible labels remain in each final reservoir.

This is bounded lifecycle/recovery evidence. It does not complete A2, approve training scope, test the formal 5M opening, price full campaigns, or establish learning strength. The report's 87.245 seconds measure allocated process wall time, not kernel-active GPU time. Differences from the earlier R1 native-decoder profile prevent a reuse-only timing comparison.
