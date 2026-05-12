# Optional Full-Dataset Smoke

`TICKET-0040` adds the first opt-in full-dataset metric path.

The training wrapper now has `--mode cached`, which resolves cached MNIST IDX
files by split or by explicit image and label paths. The mode can be bounded
with `--max-samples`, and missing cache files produce a `mnist-train-skip`
diagnostic instead of making the default e2e gate depend on downloads.

The manifest-covered `mnist-full-dataset-smoke` case builds a tiny local IDX
cache and runs cached training metrics through the same wrapper path. That keeps
the proof network-free while exercising the full-dataset control flow.
