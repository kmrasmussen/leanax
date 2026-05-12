# Exact MNIST Loss Runtime

`TICKET-0069` extends the exact forward runtime into classifier-sized loss.

`exact-mnist-loss-runtime` reuses the full `2x784 -> 8 -> 10` forward body,
adds deterministic `2x10` labels, computes row-wise softmax cross-entropy, and
returns mean loss through `mlir-runner`. The runtime result is `2.261078`.

The new `exact-mnist-loss-runtime-oracle` computes the same loss in Python and
checks the manifest expectation, keeping this distinct from the upcoming
gradient and train-step runtime tickets.
