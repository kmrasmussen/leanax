from __future__ import annotations

import json
import math
from pathlib import Path

from exact_mnist_forward_runtime_oracle import expected_forward_tensors
from exact_mnist_loss_runtime_oracle import LABELS, softmax_cross_entropy
from numeric_oracles import Tensor, elementwise, matmul, transpose_2d


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "e2e/manifest.txt"
OUTPUT = REPO / "generated/exact-mnist-gradient-runtime-oracle.json"
TOLERANCE = 5.0e-3


def reduce_last_dim(value: Tensor) -> Tensor:
    if len(value.shape) != 2:
        raise AssertionError(f"expected rank-2 tensor, got {value.shape}")
    rows, cols = value.shape
    return Tensor((rows, 1), tuple(sum(value.data[row * cols + col] for col in range(cols)) for row in range(rows)))


def gradient_tensors() -> dict[str, Tensor]:
    forward = expected_forward_tensors()
    logits = forward["logits"]
    exp_logits = Tensor(logits.shape, tuple(math.exp(value) for value in logits.data))
    denom_keepdim = reduce_last_dim(exp_logits)
    denom = Tensor(logits.shape, tuple(denom_keepdim.data[row] for row in range(2) for _ in range(10)))
    probs = elementwise(exp_logits, denom, lambda value, row_total: value / row_total)
    delta = elementwise(probs, LABELS, lambda prob, label: (prob - label) / 2.0)
    relu_mask = Tensor(forward["hidden_pre"].shape, tuple(1.0 if value > 0.0 else 0.0 for value in forward["hidden_pre"].data))
    hidden_grad = matmul(delta, transpose_2d(forward["w2"]))
    pre_activation_grad = elementwise(hidden_grad, relu_mask, lambda grad, mask: grad * mask)
    grad_w1 = matmul(transpose_2d(forward["x"]), pre_activation_grad)
    grad_b1_keepdim = reduce_last_dim(transpose_2d(pre_activation_grad))
    grad_w2 = matmul(transpose_2d(forward["hidden"]), delta)
    grad_b2_keepdim = reduce_last_dim(transpose_2d(delta))
    return {
        "loss": Tensor.scalar(softmax_cross_entropy(logits, LABELS)),
        "grad_w1": grad_w1,
        "grad_b1": Tensor((8,), grad_b1_keepdim.data),
        "grad_w2": grad_w2,
        "grad_b2": Tensor((10,), grad_b2_keepdim.data),
    }


def checksum(tensors: dict[str, Tensor]) -> float:
    values = (
        tensors["loss"].data
        + tensors["grad_w1"].data
        + tensors["grad_b1"].data
        + tensors["grad_w2"].data
        + tensors["grad_b2"].data
    )
    return sum((index + 1) * value for index, value in enumerate(values))


def manifest_expected() -> float:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[:2] == ["runtime", "exact-mnist-gradient-runtime"]:
            return float(fields[4])
    raise AssertionError("missing runtime exact-mnist-gradient-runtime manifest entry")


def verify() -> None:
    expected = checksum(gradient_tensors())
    manifested = manifest_expected()
    if abs(expected - manifested) > TOLERANCE:
        raise AssertionError(
            f"manifest gradient checksum drifted: oracle={expected:.9f} manifest={manifested:.9f}"
        )
    report = {
        "schema": "leanax.exact_mnist_gradient_runtime_oracle.v1",
        "classifier": {"batch": 2, "inputs": 784, "hidden": 8, "classes": 10},
        "checksum": expected,
        "manifest_expected": manifested,
        "tolerance": TOLERANCE,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"exact-mnist-gradient-runtime-oracle checksum={expected:.9f} output={OUTPUT}")


if __name__ == "__main__":
    verify()
