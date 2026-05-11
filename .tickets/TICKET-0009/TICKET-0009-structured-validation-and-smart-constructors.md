# TICKET-0009: Structured Validation And Smart Constructors

## Problem

Validation failures are currently string-only and many modules are assembled by
hand. That is enough for the first slice, but it will not scale to neural-network
expressions.

## Goal

Introduce structured validation errors and checked constructors that make valid
IR easier to build and invalid IR easier to test.

## E2E Focus

Every new error variant should have an expected-failure manifest case. Every
new constructor used by examples should be covered by at least one passing
golden case.

## Status

Ready for development.
