# Gray Go

Gray Go is a two-player Go variant with simultaneous moves on a toroidal board. Both players secretly choose a placement or pass; collisions create neutral gray stones.

The default board is 9×9. Black, white, and gray groups participate in captures, and fractional area scoring divides gray's contribution between the players.

## Design

[Game specification and AI training design](docs/design.md) contains the rules, proposed neural-network architecture, training and evaluation plans, and review history.

## Project status

This initial repository contains the design. The first implementation milestone is Phase A: the deterministic resolver, correctness tests, checkpoint/storage checks, and profiling described in the design document.
