# Derived Mask Train Command Wiring

`TICKET-0050` moves the command-facing classifier path onto the derived-mask
train-step artifact.

`mnist_train_command.py` now requires and reports
`generated/mnist-train-step-derived-mask.mlir`. The artifact-composition check
also requires that derived-mask artifact, while the explicit-mask
`mnist-train-step` stays manifested as a compatibility fixture.
