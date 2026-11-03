# Six original bounded Gray Go neural cases launched

Original parent 2801276 launched all three original seeds for both bounded families:

| Family | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| gray-pure | 2801277 | 2801278 | 2801279 |
| gray-mixed | 2801280 | 2801281 | 2801282 |

Each case uses the full64/eight-block CPU/features model, original one-step R4 recipe, 15M records, 500k buffers/batch 512, 12,800 fresh-Q target refresh and 5M–7M hand-off. Original source a714ee0 is imported from the qualified frozen cache package; run manifests accurately retain orchestration commit fa33155 separately.

[Actual initialization comparison](initial-model-comparison.json) verifies all six private zero-record checkpoints, empty optimizer moments, Q-equal targets, no active episode or collected roles, and distinct actual Q/average weights for every seed pair within each family. [Copied metadata](retention.json) and [permanent original trainer payload inventory](raw-initial-model-inventory.json) retain complete initial provenance outside rolling resume rotation. No public evaluation checkpoint zero is claimed.

Two immediate post-Popen command-line snapshots were empty before exec (pure seed 1 and mixed seed 2). The [separate identity confirmation](process-identity-confirmation.json) revalidates every unchanged original PID/start tick against its full exact allocated argv. The original launch job snapshot is preserved, and no worker was restarted or raw manifest rewritten. The observer binds the confirmed identities.

[Training allocation](../../experiments/nfsp-gray-neural-seeds012-allocation.json) reserves 202.5 aggregate CPU process-hours, with individual pure/mixed caps of 30.75/36.75 hours, graceful budget signals 30 seconds before each cap and hard stops before it. [Independent observer allocation](../../experiments/nfsp-gray-neural-seeds012-audit-allocation.json) fixes source plans, budgets, initial seals and case identities before final results.

Ninety million aggregate formal records are selected and running, not completed. No A2 pass, seed replacement, extension, GPU/table-control allocation, 9x9 pilot or study is claimed.
