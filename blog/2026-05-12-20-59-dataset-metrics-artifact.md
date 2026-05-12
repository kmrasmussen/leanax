# Dataset Metrics Artifact

`TICKET-0052` turns the cached dataset training sweep from console output into
a verifier-readable artifact.

The sweep now writes `generated/mnist-real-dataset-metrics.json` with schema
`leanax.mnist_dataset_metrics.v1`. The JSON records mode, split, epochs,
sample count, batch count, first and final loss, first and final accuracy, and
the generated LeanAX artifacts that the run required.

The new `mnist-dataset-metrics` manifest case checks that artifact immediately
after the cached sweep. It verifies the schema, bounded sample counts, loss
decrease, non-regressing accuracy, referenced artifact paths, and the
command-facing `generated/mnist-train-step-derived-mask.mlir` artifact.
