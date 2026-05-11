# TICKET-0013: JAX-Shaped LeanAX DSL

## Goal

Design the first DSL layer where examples look like pure tensor functions with
explicit params, batches, and transform boundaries.

## E2E Focus

A two-layer MLP forward pass should lower through the same e2e runner as raw IR
examples.

## Status

Ready for development.
