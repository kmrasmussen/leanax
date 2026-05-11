from __future__ import annotations

import sys


def loss(weight: float, xs: list[float], ys: list[float]) -> float:
    return sum((weight * x - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


def grad(weight: float, xs: list[float], ys: list[float]) -> float:
    return sum(2.0 * x * (weight * x - y) for x, y in zip(xs, ys)) / len(xs)


def main() -> int:
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [2.0 * x for x in xs]
    weight = 0.0
    lr = 0.1
    first = loss(weight, xs, ys)

    for _ in range(20):
        weight += -lr * grad(weight, xs, ys)

    final = loss(weight, xs, ys)
    if not final < first * 0.01:
        print(f"loss did not decrease enough: first={first} final={final}", file=sys.stderr)
        return 1
    print(f"training-loop first_loss={first:.6f} final_loss={final:.6f} weight={weight:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
