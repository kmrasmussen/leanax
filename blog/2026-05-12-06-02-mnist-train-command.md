# MNIST Train Command Wrapper

`TICKET-0038` gives the fixture classifier path a stable operator-facing
wrapper.

The new `mnist_train_command.py --mode fixture` command checks the generated
forward, loss, gradient, parameter update, and monolithic train-step artifacts
before running the deterministic ten-class fixture training loop. It prints
stable metric fields for mode, epochs, samples, batches, loss, accuracy, and
artifact paths.

This keeps the implementation honest: the command still uses host Python for
loop orchestration, but it refuses to run without the checked LeanAX compiler
artifacts that define the current classifier path.
