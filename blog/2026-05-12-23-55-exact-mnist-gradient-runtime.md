# Exact MNIST Gradient Runtime

`TICKET-0070` adds the full-size backward runtime checksum.

`exact-mnist-gradient-runtime` reuses the exact forward and loss body, derives
the ReLU mask from `hidden_pre`, computes `grad_w2`, `grad_b2`, `grad_w1`, and
`grad_b1`, then returns a weighted checksum over loss plus all gradients.

The generated golden is 72,125 lines and returns `-131.4983` through
`mlir-runner`. `exact-mnist-gradient-runtime-oracle` computes the same checksum
from the Python oracle tensors with an explicit `5e-3` tolerance for the long
f32 accumulation.
