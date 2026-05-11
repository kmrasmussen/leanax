# MNIST Parameter Tree

`TICKET-0029` is complete.

The optimizer artifact now covers the full MNIST classifier parameter set:
`w1`, `b1`, `w2`, and `b2`. The new `mnist-parameter-tree` module accepts the
four parameters plus matching gradients, broadcasts the learning-rate scalar to
each shape, and returns `next_w1`, `next_b1`, `next_w2`, and `next_b2`.

The e2e runner checks the generated multi-output module with golden text, MLIR
generic parsing, lowering manifest validation, and a Python oracle that compares
every returned tensor. A new expected-failure case catches mismatched classifier
bias-gradient shape.
