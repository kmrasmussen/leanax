# TICKET-0001: Planning And Ticket Scaffold

## Problem

LeanAX has early notes but no durable planning surface or ticket backlog.

## Goal

Create `amitious_plan/` and `.tickets/` in the same broad on-disk style used by
the local `dottickets`, `leanix`, and `nixparserlean` projects.

## In Scope

- Planning documents that describe the ambitious roadmap.
- Dot-ticket status settings.
- Phase-one tickets that are small enough to implement.

## Acceptance Criteria

1. `amitious_plan/README.md` links the planning documents.
2. `.tickets/README.md` explains the backlog policy.
3. `.tickets/.ticket/settings/statuses.json` and schema exist.
4. Phase-one tickets exist as numbered directories with `ticket-state.json`.

## First Slice

Create the roadmap documents and five concrete phase-one tickets.
