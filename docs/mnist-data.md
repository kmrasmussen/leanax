# MNIST Data

LeanAX keeps data loading outside the trusted Lean core.

The current e2e gate uses a deterministic MNIST-shaped fixture in
`e2e/python/mnist_fixture.py`. It generates four samples with flattened
`28 * 28` image vectors, normalizes pixel values into `[0, 1]`, batches them in
pairs, and represents labels as one-hot vectors with length `10`.

This fixture is intentionally not the full MNIST dataset. It exists so local and
flake checks can verify shape, dtype-like normalization, label range, batching,
and determinism without a network dependency.

## Full IDX Route

The optional full-dataset route starts from canonical MNIST IDX files and uses
the same host-side contract as the fixture:

1. download or locate MNIST outside Lean,
2. cache it in a reproducible host location,
3. normalize images to flattened f32-compatible vectors,
4. encode labels in the loss-compatible one-hot format,
5. pass batches to checked LeanAX-compiled steps.

`e2e/python/mnist_fixture.py` exposes `load_idx_files(images_path, labels_path)`
for direct local runs and `load_mnist_split(split, cache_dir=...)` for cache
based train/test loading. It expects IDX image files with magic `2051`, IDX
label files with magic `2049`, `28x28` images, labels in `[0, 9]`, matching
image/label counts, and a sample count divisible by the static batch size.

Suggested local cache layout:

```text
$XDG_CACHE_HOME/leanax/mnist/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

If `XDG_CACHE_HOME` is unset, use `~/.cache/leanax/mnist`. Downloading stays
outside the default e2e gate: fetch the dataset with a normal host tool, place
or symlink the four IDX files into that cache, then pass the image and label
paths to the local classifier runner that consumes `load_idx_files`.

The resolver recognizes two split names:

- `train`: `train-images-idx3-ubyte` plus `train-labels-idx1-ubyte`
- `test`: `t10k-images-idx3-ubyte` plus `t10k-labels-idx1-ubyte`

Explicit image/label paths may be passed instead of a cache split, but both
paths must be provided together.

Expected failure modes are explicit: missing files fail at path read time, bad
magic numbers reject non-IDX inputs, non-`28x28` images reject incompatible
datasets, mismatched counts reject corrupt image/label pairs, out-of-range
labels reject non-MNIST targets, and non-divisible sample counts reject batches
that would not match the current static `2x784` classifier artifact shape.

The manifest cases `mnist-idx-sample` and `mnist-cache-resolver` exercise this
code path with tiny local IDX bytes. That keeps fixture mode as the default
smoke path and keeps the full e2e gate network-free while still checking the
parser, resolver, split names, partial-cache diagnostics, and batch contract used
by the full dataset route.
