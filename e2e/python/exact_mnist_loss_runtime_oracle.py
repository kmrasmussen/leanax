from __future__ import annotations

import json
import math
from pathlib import Path

from exact_mnist_forward_runtime_oracle import expected_forward_logits
from numeric_oracles import Tensor, tensor


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "e2e/manifest.txt"
OUTPUT = REPO / "generated/exact-mnist-loss-runtime-oracle.json"


LABELS = tensor(
    (2, 10),
    [
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ],
)


def softmax_cross_entropy(logits: Tensor, labels: Tensor) -> float:
    if logits.shape != (2, 10) or labels.shape != (2, 10):
        raise AssertionError(f"unexpected shapes: logits={logits.shape} labels={labels.shape}")
    losses = []
    for row in range(2):
        row_values = logits.data[row * 10 : (row + 1) * 10]
        row_labels = labels.data[row * 10 : (row + 1) * 10]
        exp_values = [math.exp(value) for value in row_values]
        denom = sum(exp_values)
        selected = [
            label * math.log(exp_value / denom)
            for label, exp_value in zip(row_labels, exp_values)
        ]
        losses.append(-sum(selected))
    return sum(losses) / 2.0


def manifest_expected() -> float:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[:2] == ["runtime", "exact-mnist-loss-runtime"]:
            return float(fields[4])
    raise AssertionError("missing runtime exact-mnist-loss-runtime manifest entry")


def verify() -> None:
    expected = softmax_cross_entropy(expected_forward_logits(), LABELS)
    manifested = manifest_expected()
    if abs(expected - manifested) > 1.0e-4:
        raise AssertionError(
            f"manifest loss drifted: oracle={expected:.9f} manifest={manifested:.9f}"
        )
    report = {
        "schema": "leanax.exact_mnist_loss_runtime_oracle.v1",
        "classifier": {"batch": 2, "inputs": 784, "hidden": 8, "classes": 10},
        "loss": expected,
        "manifest_expected": manifested,
        "tolerance": 1.0e-4,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"exact-mnist-loss-runtime-oracle loss={expected:.9f} output={OUTPUT}")


if __name__ == "__main__":
    verify()
