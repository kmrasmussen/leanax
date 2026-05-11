# MNIST Fixture Loader

`TICKET-0025` is complete.

LeanAX now has a host-side MNIST-shaped fixture path. The fixture is deliberately
small and deterministic: four generated samples, flattened `28x28` image
vectors, pixel values normalized to `[0, 1]`, batch size `2`, and one-hot labels
with `10` classes.

The e2e manifest now has a dedicated `data-loader` outcome, so data checks are
not hidden under numeric kernel or training-loop cases. The fixture check proves
batch shape, label shape, label range, normalization, and deterministic
construction without downloading the full dataset.

The full MNIST path remains future work, but the host-side contract is now
explicit and documented in `docs/mnist-data.md`.
