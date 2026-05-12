# MNIST Train-Step Artifact

`TICKET-0032` is complete.

The first train-step slice is deliberately stitched in the e2e layer. The new
`mnist_train_step_artifact.py` check requires the generated forward, loss,
final-layer gradient, first-layer gradient, and parameter-tree update artifacts
to exist. It then computes one deterministic fixture batch update with the same
math and checks that loss decreases and the parameter update is non-zero.

This is not yet a single monolithic Lean train-step module. It is the first
checked integration point where the compiler artifacts required by a classifier
train step are present together and guarded by the normal e2e manifest.
