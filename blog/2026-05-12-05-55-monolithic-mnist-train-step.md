# Monolithic MNIST Train Step

`TICKET-0036` and `TICKET-0037` move the classifier path from a Python-composed
set of artifacts to one fixed-shape LeanAX train-step artifact.

The new `mnist-train-step` case lowers the forward pass, ten-class softmax
cross entropy, final-layer gradient, first-layer ReLU gradient, and SGD update
in one module. It returns the full updated parameter tree plus the batch loss,
and the numeric oracle compares the result to the same deterministic classifier
math used by the existing e2e smoke.

The first version still accepts `relu_mask` as an input. That keeps the current
primitive boundary honest: deriving the mask inside the module belongs with a
future compare/select slice. The validation suite now has train-step-specific
negative cases for labels, hidden masks, and parameter shapes.
