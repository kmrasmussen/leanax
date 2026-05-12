from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from mnist_fixture import split_idx_paths
from mnist_idx_sample import sample_images, sample_labels
from mnist_train_command import cached_metrics, render_metrics, require_checked_artifacts


def write_tiny_cache(root: Path) -> None:
    images_path, labels_path = split_idx_paths("train", root)
    images_path.write_bytes(sample_images())
    labels_path.write_bytes(sample_labels())


def verify() -> None:
    artifacts = require_checked_artifacts()
    with TemporaryDirectory(prefix="leanax-mnist-full-smoke-") as tmp:
        root = Path(tmp)
        write_tiny_cache(root)
        metrics = cached_metrics(
            split="train",
            epochs=2,
            cache_dir=str(root),
            images_idx=None,
            labels_idx=None,
            max_samples=4,
        )

    if metrics.mode != "cached-train":
        raise AssertionError(f"unexpected mode {metrics.mode}")
    if metrics.samples != 4 or metrics.batches != 2 or metrics.epochs != 2:
        raise AssertionError(f"unexpected cached metrics shape: {metrics}")
    if not metrics.final_loss < metrics.first_loss:
        raise AssertionError(f"cached smoke loss did not decrease: {metrics.first_loss} -> {metrics.final_loss}")
    if metrics.final_accuracy < metrics.first_accuracy:
        raise AssertionError(
            f"cached smoke accuracy regressed: {metrics.first_accuracy} -> {metrics.final_accuracy}"
        )

    print(render_metrics(metrics, artifacts))


if __name__ == "__main__":
    verify()
