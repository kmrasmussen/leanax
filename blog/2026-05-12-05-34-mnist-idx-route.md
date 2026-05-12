# MNIST IDX Route

`TICKET-0034` closes the gap between the deterministic MNIST-shaped fixture and
real MNIST files without putting the network on the critical e2e path.

The loader now parses canonical IDX image and label bytes, checks magic numbers,
requires `28x28` images, rejects mismatched counts and out-of-range labels, and
returns the same `MnistBatch` contract as the fixture: flattened normalized
image vectors and one-hot labels.

The manifest case `mnist-idx-sample` builds a four-sample IDX dataset in memory.
That gives the full-dataset route an e2e check while keeping normal CI and
flake runs hermetic.
