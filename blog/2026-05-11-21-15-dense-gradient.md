# Dense Gradient Slice

`TICKET-0023` is complete.

The new `grad-dense-loss` case is the first gradient artifact that goes through
a dense layer. It targets a tiny one-layer scalar loss:

```text
loss = sum((x @ w + b)^2)
grad_w = x^T @ (2 * (x @ w + b))
```

This is still a restricted generated-gradient path rather than a full reverse
mode system, but it exercises the key pieces needed next: matmul, bias
broadcast, scalar-loss residuals, transpose, and a matrix gradient result.

The e2e gate compares the generated `grad_w` against a Python analytic oracle.
It also includes `bad-grad-dense-shape` so unsupported gradient shapes fail
before lowering.
