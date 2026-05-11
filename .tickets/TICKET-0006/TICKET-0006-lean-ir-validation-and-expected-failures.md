# TICKET-0006: Lean IR Validation And Expected Failures

## Problem

Golden rendering proves stable output for valid examples, but it does not prove
LeanAX rejects malformed tensor IR before lowering.

## Goal

Add a Lean validation pass and wire one expected failure into the Rust e2e
runner.

## In Scope

- Defined-reference checks.
- Duplicate-result checks.
- Elementwise operand/result type checks.
- Dot-general shape checks for rank-2 matmul.
- An invalid e2e manifest with expected stderr text.

## Acceptance Criteria

1. Valid golden cases still pass.
2. `bad-add-shape` fails before writing a module.
3. The Rust e2e runner verifies the expected failure text.

## First Slice

Implement enough validation to catch mismatched elementwise shapes.
