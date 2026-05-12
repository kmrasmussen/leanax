# TICKET-0070: Exact-Shape Derived-Mask Gradient Runtime Checksum

## Problem

Direct MNIST runtime requires backward-path evidence, not only forward and loss
checksums. The derived ReLU mask and dense gradients are the highest-risk scale
step before the full train-step update.

## Goal

Generate and execute an exact-shape derived-mask gradient runtime checksum.

## In Scope

- Reuse exact-shape forward and loss generation.
- Derive the ReLU mask internally.
- Generate gradients for `w2`, `b2`, `w1`, and `b1`.
- Return deterministic checksums over loss and gradients.
- Compare against the existing oracle family.

## E2E Focus

Run the exact-shape gradient checksum through the external runtime path, or
record a precise budget blocker.

## Acceptance Criteria

1. Exact-shape gradient runtime evidence exists.
2. The checksum covers both layers and biases.
3. Oracle comparison is deterministic and tolerance is explicit.
4. The full Nix e2e gate passes.

## First Slice

Prefer checksum coverage over returning every scalar. The artifact should prove
gradient semantics without bloating the runtime output ABI.

## Status

Ready for analysis.
