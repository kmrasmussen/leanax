# TICKET-0010: Neural-Network Primitive Ops

## Goal

Add enough primitive operations to express small MLP pieces: constants,
broadcast, reshape, transpose, and reduce-sum.

## E2E Focus

Each primitive needs at least one passing golden module and at least one relevant
validation-failure case when shape or dtype rules can be violated.

## Status

Ready for development.
