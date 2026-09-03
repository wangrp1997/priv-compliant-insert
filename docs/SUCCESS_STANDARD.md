# Force-enter success standard (locked)

Success requires **all**:

1. Deployable force (or ConnTact height) hole enter
2. Tip on tray/hole face during spiral (reject float-over-mouth via along/seat)
3. Peg axis roughly upright at enter (`axis_err ≤ 25°`)

Reject explicitly:

- Sideways / fallen peg (e.g. ep02 → `force_hole_axis_reject`)
- Floating tip over hole with lat_min only (e.g. ep03 → `force_hole_along_reject`)

Hard audit knobs (`configs/scheme_l3/S2_force_insert.yaml`):

- `surface_planned_priv_enter_max_axis_err_deg: 25`
- `surface_planned_priv_enter_max_along_m: 0.100`
- `surface_planned_priv_enter_max_float_m: 0.004`
- `surface_planned_priv_enter_max_tip_rise_m: 0.006`

`spiral_contact_frac` is diagnostic by default (`min_contact_frac: 0`).
