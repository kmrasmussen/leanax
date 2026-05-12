from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mnist_classifier_smoke import EPOCHS, train_batches
from mnist_fixture import fixture_batches
from mnist_train_step_artifact import require_artifact


REQUIRED_ARTIFACTS = {
    "mnist_forward": ("generated/mnist-forward.mlir", ["tensor<2x784xf32>", "tensor<2x10xf32>"]),
    "mnist_cross_entropy": ("generated/mnist-cross-entropy.mlir", ["stablehlo.exponential", "value = \"-0.5\""]),
    "grad_softmax_dense": ("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"]),
    "grad_relu_dense": ("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1"]),
    "mnist_parameter_tree": ("generated/mnist-parameter-tree.mlir", ["%next_w1", "%next_b1", "%next_w2", "%next_b2"]),
    "mnist_train_step": ("generated/mnist-train-step.mlir", ["%next_w1", "%next_b1", "%next_w2", "%next_b2", "%loss"]),
}


@dataclass(frozen=True)
class Metrics:
    mode: str
    epochs: int
    samples: int
    batches: int
    first_loss: float
    final_loss: float
    first_accuracy: float
    final_accuracy: float


def require_checked_artifacts() -> list[str]:
    paths = []
    for _name, (path, required_text) in REQUIRED_ARTIFACTS.items():
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"missing generated artifact {path}; run the full e2e gate or emit the manifested numeric cases first"
            )
        require_artifact(path, required_text)
        paths.append(path)
    return paths


def fixture_metrics(epochs: int) -> Metrics:
    batches = fixture_batches()
    first_loss, final_loss, first_accuracy, final_accuracy = train_batches(batches, epochs)
    sample_count = sum(len(batch.images) for batch in batches)
    return Metrics(
        mode="fixture",
        epochs=epochs,
        samples=sample_count,
        batches=len(batches),
        first_loss=first_loss,
        final_loss=final_loss,
        first_accuracy=first_accuracy,
        final_accuracy=final_accuracy,
    )


def render_metrics(metrics: Metrics, artifacts: list[str]) -> str:
    artifact_text = ",".join(artifacts)
    return (
        "mnist-train "
        f"mode={metrics.mode} "
        f"epochs={metrics.epochs} "
        f"samples={metrics.samples} "
        f"batches={metrics.batches} "
        f"first_loss={metrics.first_loss:.6f} "
        f"final_loss={metrics.final_loss:.6f} "
        f"first_accuracy={metrics.first_accuracy:.2f} "
        f"final_accuracy={metrics.final_accuracy:.2f} "
        f"artifacts={artifact_text}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LeanAX MNIST classifier training wrapper.")
    parser.add_argument("--mode", choices=["fixture"], default="fixture")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")

    artifacts = require_checked_artifacts()
    metrics = fixture_metrics(args.epochs)
    if not metrics.final_loss < metrics.first_loss:
        raise AssertionError(f"classifier loss did not decrease: {metrics.first_loss} -> {metrics.final_loss}")
    if metrics.final_accuracy < metrics.first_accuracy:
        raise AssertionError(f"classifier accuracy regressed: {metrics.first_accuracy} -> {metrics.final_accuracy}")
    print(render_metrics(metrics, artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
