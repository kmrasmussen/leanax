# Ten-Class MNIST Smoke

`TICKET-0033` is complete.

The old MNIST smoke stayed deliberately tiny: two features and parity labels.
The new `mnist_classifier_smoke.py` command uses the actual classifier shape:
flattened `28x28` fixture images, hidden width `8`, and ten output classes.

The command checks that the generated LeanAX forward, ten-class loss,
softmax-dense gradient, ReLU first-layer gradient, and full parameter-tree
update artifacts exist. It then runs a deterministic short host loop and asserts
that loss improves without accuracy regression. The e2e manifest covers it as a
training-loop case.
