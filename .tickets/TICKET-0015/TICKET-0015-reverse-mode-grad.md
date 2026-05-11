# TICKET-0015: Reverse-Mode Grad For Scalar Losses

## Goal

Generate backward programs for a restricted scalar-loss subset.

## E2E Focus

Gradient cases should be checked against finite differences or a Python/JAX
oracle on tiny examples.

## Status

Ready for development.
