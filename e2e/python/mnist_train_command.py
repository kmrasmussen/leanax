from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from mnist_classifier_smoke import EPOCHS, train_batches
from mnist_fixture import MnistBatch, fixture_batches, load_mnist_split
from mnist_train_step_artifact import require_artifact


REQUIRED_ARTIFACTS = {
    "mnist_forward": ("generated/mnist-forward.mlir", ["tensor<2x784xf32>", "tensor<2x10xf32>"]),
    "mnist_cross_entropy": ("generated/mnist-cross-entropy.mlir", ["stablehlo.exponential", "value = \"-0.5\""]),
    "grad_softmax_dense": ("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"]),
    "grad_relu_dense": ("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1"]),
    "mnist_parameter_tree": ("generated/mnist-parameter-tree.mlir", ["%next_w1", "%next_b1", "%next_w2", "%next_b2"]),
    "mnist_train_step_derived_mask": (
        "generated/mnist-train-step-derived-mask.mlir",
        ["stablehlo.compare", "stablehlo.select", "%relu_mask", "%next_w1", "%loss"],
    ),
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


class OptionalDatasetMissing(Exception):
    pass


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


def limit_batches(batches: list[MnistBatch], max_samples: int | None) -> list[MnistBatch]:
    if max_samples is None:
        return batches
    if max_samples <= 0:
        raise ValueError(f"--max-samples must be positive, got {max_samples}")
    selected = []
    seen = 0
    for batch in batches:
        if seen + len(batch.images) > max_samples:
            break
        selected.append(batch)
        seen += len(batch.images)
    if not selected:
        raise ValueError(f"--max-samples={max_samples} is smaller than one complete batch")
    return selected


def metrics_for_batches(mode: str, batches: list[MnistBatch], epochs: int) -> Metrics:
    first_loss, final_loss, first_accuracy, final_accuracy = train_batches(batches, epochs)
    sample_count = sum(len(batch.images) for batch in batches)
    return Metrics(
        mode=mode,
        epochs=epochs,
        samples=sample_count,
        batches=len(batches),
        first_loss=first_loss,
        final_loss=final_loss,
        first_accuracy=first_accuracy,
        final_accuracy=final_accuracy,
    )


def fixture_metrics(epochs: int) -> Metrics:
    return metrics_for_batches("fixture", fixture_batches(), epochs)


def cached_metrics(
    split: str,
    epochs: int,
    cache_dir: str | None,
    images_idx: str | None,
    labels_idx: str | None,
    max_samples: int | None,
) -> Metrics:
    try:
        batches = load_mnist_split(
            split=split,
            cache_dir=cache_dir,
            images_path=images_idx,
            labels_path=labels_idx,
        )
    except FileNotFoundError as err:
        raise OptionalDatasetMissing(str(err)) from err
    return metrics_for_batches(f"cached-{split}", limit_batches(batches, max_samples), epochs)


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
    parser.add_argument("--mode", choices=["fixture", "cached"], default="fixture")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--cache-dir")
    parser.add_argument("--images-idx")
    parser.add_argument("--labels-idx")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")

    artifacts = require_checked_artifacts()
    try:
        if args.mode == "fixture":
            metrics = fixture_metrics(args.epochs)
        else:
            metrics = cached_metrics(
                args.split,
                args.epochs,
                args.cache_dir,
                args.images_idx,
                args.labels_idx,
                args.max_samples,
            )
    except OptionalDatasetMissing as err:
        print(f"mnist-train-skip mode=cached split={args.split} reason=missing-cache detail={err}")
        return 0

    if not metrics.final_loss < metrics.first_loss:
        raise AssertionError(f"classifier loss did not decrease: {metrics.first_loss} -> {metrics.final_loss}")
    if metrics.final_accuracy < metrics.first_accuracy:
        raise AssertionError(f"classifier accuracy regressed: {metrics.first_accuracy} -> {metrics.final_accuracy}")
    print(render_metrics(metrics, artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
