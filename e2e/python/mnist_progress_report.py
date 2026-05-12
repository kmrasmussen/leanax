from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "e2e/manifest.txt"


EXPECTED = {
    "affine_external_runtime": True,
    "artifact_composed_train_step": True,
    "direct_mnist_external_runtime": False,
    "fixture_only_default": True,
    "full_dataset_training": False,
    "idx_full_dataset_loader": True,
    "monolithic_mnist_train_step": False,
    "mnist_cross_entropy_artifact": True,
    "mnist_forward_artifact": True,
    "mnist_parameter_tree_artifact": True,
    "relu_dense_gradient_artifact": True,
    "softmax_dense_gradient_artifact": True,
    "ten_class_fixture_training": True,
}


def manifest_entries() -> set[tuple[str, str]]:
    entries = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            entries.add((fields[0], fields[1]))
    return entries


def artifact_contains(path: str, required_text: list[str]) -> bool:
    artifact = REPO / path
    if not artifact.is_file():
        return False
    text = artifact.read_text(encoding="utf-8")
    return all(required in text for required in required_text)


def report() -> dict[str, bool]:
    entries = manifest_entries()
    return {
        "affine_external_runtime": ("runtime", "affine-runtime") in entries,
        "artifact_composed_train_step": (
            ("training-loop", "mnist-train-step-artifact") in entries
            and artifact_contains("generated/mnist-forward.mlir", ["tensor<2x784xf32>"])
            and artifact_contains("generated/mnist-cross-entropy.mlir", ["stablehlo.exponential"])
            and artifact_contains("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"])
            and artifact_contains("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1"])
            and artifact_contains("generated/mnist-parameter-tree.mlir", ["%next_w1", "%next_b1"])
        ),
        "direct_mnist_external_runtime": False,
        "fixture_only_default": ("data-loader", "mnist-fixture") in entries,
        "full_dataset_training": False,
        "idx_full_dataset_loader": ("data-loader", "mnist-idx-sample") in entries,
        "monolithic_mnist_train_step": ("numeric", "mnist-train-step") in entries,
        "mnist_cross_entropy_artifact": (
            ("numeric", "mnist-cross-entropy") in entries
            and artifact_contains(
                "generated/mnist-cross-entropy.mlir",
                ["stablehlo.exponential", "tensor<2x10xf32>"],
            )
        ),
        "mnist_forward_artifact": (
            ("numeric", "mnist-forward") in entries
            and artifact_contains(
                "generated/mnist-forward.mlir",
                ["tensor<2x784xf32>", "stablehlo.maximum", "tensor<2x10xf32>"],
            )
        ),
        "mnist_parameter_tree_artifact": (
            ("numeric", "mnist-parameter-tree") in entries
            and artifact_contains(
                "generated/mnist-parameter-tree.mlir",
                ["%next_w1", "%next_b1", "%next_w2", "%next_b2"],
            )
        ),
        "relu_dense_gradient_artifact": (
            ("numeric", "grad-relu-dense") in entries
            and artifact_contains("generated/grad-relu-dense.mlir", ["%grad_w1", "%grad_b1", "%relu_mask"])
        ),
        "softmax_dense_gradient_artifact": (
            ("numeric", "grad-softmax-dense") in entries
            and artifact_contains("generated/grad-softmax-dense.mlir", ["%grad_w2", "%grad_b2"])
        ),
        "ten_class_fixture_training": ("training-loop", "mnist-classifier-smoke") in entries,
    }


def main() -> int:
    actual = report()
    if actual != EXPECTED:
        raise AssertionError(
            "MNIST progress report drifted:\n"
            f"expected={json.dumps(EXPECTED, sort_keys=True)}\n"
            f"actual={json.dumps(actual, sort_keys=True)}"
        )

    print("mnist-progress " + json.dumps(actual, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
