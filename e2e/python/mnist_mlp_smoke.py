from __future__ import annotations

import math
from pathlib import Path

from mnist_fixture import fixture_batches


LEARNING_RATE = 3.0


def require_artifact(path: str, required_ops: list[str]) -> None:
    text = Path(path).read_text(encoding="utf-8")
    for op in required_ops:
        if op not in text:
            raise AssertionError(f"{path} does not contain {op}")


def features(image: list[float]) -> tuple[float, float]:
    return (image[0], image[1])


def label_parity(one_hot: list[float]) -> int:
    return one_hot.index(1.0) % 2


def logits(weights: list[list[float]], bias: list[float], x: tuple[float, float]) -> list[float]:
    return [
        x[0] * weights[0][col] + x[1] * weights[1][col] + bias[col]
        for col in range(2)
    ]


def softmax(values: list[float]) -> list[float]:
    shifted = [value - max(values) for value in values]
    exps = [math.exp(value) for value in shifted]
    denom = sum(exps)
    return [value / denom for value in exps]


def loss_and_accuracy(weights: list[list[float]], bias: list[float]) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    count = 0
    for batch in fixture_batches():
        for image, label in zip(batch.images, batch.labels):
            target = label_parity(label)
            probs = softmax(logits(weights, bias, features(image)))
            total_loss -= math.log(probs[target])
            predicted = 0 if probs[0] >= probs[1] else 1
            correct += 1 if predicted == target else 0
            count += 1
    return total_loss / count, correct / count


def train() -> tuple[float, float, float, float]:
    weights = [[0.0, 0.0], [0.0, 0.0]]
    bias = [0.0, 0.0]
    first_loss, first_accuracy = loss_and_accuracy(weights, bias)

    for _epoch in range(8):
        for batch in fixture_batches():
            grad_w = [[0.0, 0.0], [0.0, 0.0]]
            grad_b = [0.0, 0.0]
            for image, label in zip(batch.images, batch.labels):
                x = features(image)
                target = label_parity(label)
                probs = softmax(logits(weights, bias, x))
                for col in range(2):
                    delta = probs[col] - (1.0 if col == target else 0.0)
                    grad_w[0][col] += x[0] * delta
                    grad_w[1][col] += x[1] * delta
                    grad_b[col] += delta
            scale = 1.0 / len(batch.images)
            for row in range(2):
                for col in range(2):
                    weights[row][col] -= LEARNING_RATE * grad_w[row][col] * scale
            for col in range(2):
                bias[col] -= LEARNING_RATE * grad_b[col] * scale

    final_loss, final_accuracy = loss_and_accuracy(weights, bias)
    return first_loss, final_loss, first_accuracy, final_accuracy


def main() -> int:
    require_artifact("generated/cross-entropy-loss.mlir", ["stablehlo.exponential", "stablehlo.log"])
    require_artifact("generated/sgd-parameter-tree.mlir", ["%next_w", "%next_b"])

    first_loss, final_loss, first_accuracy, final_accuracy = train()
    if final_loss >= first_loss:
        raise AssertionError(f"loss did not decrease: {first_loss} -> {final_loss}")
    if final_accuracy < first_accuracy:
        raise AssertionError(f"accuracy regressed: {first_accuracy} -> {final_accuracy}")
    print(
        "mnist-mlp-smoke "
        f"first_loss={first_loss:.6f} final_loss={final_loss:.6f} "
        f"first_accuracy={first_accuracy:.2f} final_accuracy={final_accuracy:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
