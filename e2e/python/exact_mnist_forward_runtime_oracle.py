from __future__ import annotations

import json
from pathlib import Path

from numeric_oracles import Tensor, broadcast_to, elementwise, matmul, patterned_tensor


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "e2e/manifest.txt"
OUTPUT = REPO / "generated/exact-mnist-forward-runtime-oracle.json"


def checksum(values: Tensor) -> float:
    return sum((index + 1) * value for index, value in enumerate(values.data))


def expected_forward_checksum() -> float:
    x = patterned_tensor((2, 784), 0.01, 1)
    w1 = patterned_tensor((784, 8), 0.002, 3)
    b1 = patterned_tensor((8,), 0.01, 5)
    w2 = patterned_tensor((8, 10), 0.02, 7)
    b2 = patterned_tensor((10,), 0.01, 9)
    hidden_pre = elementwise(matmul(x, w1), broadcast_to(b1, (2, 8)), lambda a, b: a + b)
    hidden = elementwise(hidden_pre, broadcast_to(Tensor.scalar(0.0), (2, 8)), max)
    logits = elementwise(matmul(hidden, w2), broadcast_to(b2, (2, 10)), lambda a, b: a + b)
    return checksum(logits)


def manifest_expected() -> float:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[:2] == ["runtime", "exact-mnist-forward-runtime"]:
            return float(fields[4])
    raise AssertionError("missing runtime exact-mnist-forward-runtime manifest entry")


def verify() -> None:
    expected = expected_forward_checksum()
    manifested = manifest_expected()
    if abs(expected - manifested) > 1.0e-4:
        raise AssertionError(
            f"manifest checksum drifted: oracle={expected:.9f} manifest={manifested:.9f}"
        )
    report = {
        "schema": "leanax.exact_mnist_forward_runtime_oracle.v1",
        "classifier": {"batch": 2, "inputs": 784, "hidden": 8, "classes": 10},
        "checksum": expected,
        "manifest_expected": manifested,
        "tolerance": 1.0e-4,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"exact-mnist-forward-runtime-oracle checksum={expected:.9f} output={OUTPUT}")


if __name__ == "__main__":
    verify()
