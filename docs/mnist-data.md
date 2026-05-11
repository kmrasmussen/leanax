# MNIST Data

LeanAX keeps data loading outside the trusted Lean core.

The current e2e gate uses a deterministic MNIST-shaped fixture in
`e2e/python/mnist_fixture.py`. It generates four samples with flattened
`28 * 28` image vectors, normalizes pixel values into `[0, 1]`, batches them in
pairs, and represents labels as one-hot vectors with length `10`.

This fixture is intentionally not the full MNIST dataset. It exists so local and
flake checks can verify shape, dtype-like normalization, label range, batching,
and determinism without a network dependency.

A full dataset path should use the same host-side contract:

1. download or locate MNIST outside Lean,
2. cache it in a reproducible host location,
3. normalize images to flattened f32-compatible vectors,
4. encode labels in the loss-compatible one-hot format,
5. pass batches to checked LeanAX-compiled steps.

The future full-dataset runner should keep this fixture mode as the default e2e
smoke path and make full MNIST opt-in.
