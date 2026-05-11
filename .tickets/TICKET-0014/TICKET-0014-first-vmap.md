# TICKET-0014: First VMap Transform

## Goal

Implement the first `vmap` transform for a small pointwise subset, then extend
toward matmul and broadcast patterns.

## E2E Focus

The runner should compare transformed modules against manually batched golden
modules or numeric oracles.

## Status

Ready for development.
