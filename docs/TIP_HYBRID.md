# Tip-Hybrid method (non-rigid grasp)

## Goal

One controller for all v8 failure modes:

- float search (ep03) → **SEAT**
- tipped peg (ep02/05/07/08) → **UPRIGHT**
- tip fly / lost contact (ep09) → **SEAT** then recover
- then **SPIRAL** tip track to hole

## Priority FSM

```
SEAT  → tip on tray (force + plane), freeze spiral advance
UPRIGHT → tip-pivot rotate peg∥hole about tip, freeze advance
SPIRAL → planar_C tip spiral + light orient maintain + press
```

Refs reused:

- ConnTact: seeking force + Z compliance + align before spiral
- Craig hybrid force/position: allocate DOF (n=force, plane=pos, R=orient)
- Kim extrinsic contact: pivot about tip contact

## Deploy path

Sim seat uses peg–tray contact oracle; replace with fingertip tactile normal.
Orient uses peg/hole axes in sim; deploy: tactile extrinsic + wrist F/T.
