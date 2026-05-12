# MNIST Cache Resolver

`TICKET-0039` turns the IDX parser into a usable full-dataset route without
making the default e2e gate depend on the network.

The host loader now resolves canonical train/test split names from the documented
cache layout, accepts explicit image and label paths, and returns the same
batched contract as the deterministic fixture. The new `mnist-cache-resolver`
manifest case builds a tiny local IDX cache, loads both splits, checks explicit
paths, and verifies partial-cache diagnostics.
