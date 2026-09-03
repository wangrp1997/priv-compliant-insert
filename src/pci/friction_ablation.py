"""Runtime MuJoCo friction scaling for peg–tray contact ablation.

Privileged diagnostic only: does not change default XML assets.
Scales ``model.geom_friction`` on collision geoms after env load.
Default: scale tray/socket only (keep peg friction for grasp).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexjoco.sim.envs.assembly_geometry import names_from_raw


def _body_collision_geom_ids(model, body_id: int) -> list[int]:
    """All geoms under body subtree with contype/conaffinity contact enabled."""
    bodies = {int(body_id)}
    changed = True
    while changed:
        changed = False
        for bid in range(int(model.nbody)):
            parent = int(model.body_parentid[bid])
            if parent in bodies and bid not in bodies:
                bodies.add(bid)
                changed = True
    out: list[int] = []
    for gid in range(int(model.ngeom)):
        if int(model.geom_bodyid[gid]) not in bodies:
            continue
        # Skip visual-only (contype=0 and conaffinity=0).
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        out.append(int(gid))
    return out


def apply_friction_ablation(raw_env, cfg: dict) -> dict[str, Any]:
    """Scale peg/tray geom friction from ``sim.friction_ablation`` cfg.

    Returns a small meta dict for logging / summaries.
    """
    sim = cfg.get("sim") or {}
    fa = sim.get("friction_ablation") or {}
    if not bool(fa.get("enable", False)):
        return {"enabled": False}

    peg_scale = float(fa.get("peg_scale", 1.0))
    tray_scale = float(fa.get("tray_scale", 0.25))
    scale_all_components = bool(fa.get("scale_all_components", True))
    # Components: [sliding, torsional, rolling]
    components = fa.get("components")
    if components is None:
        components = [True, True, True] if scale_all_components else [True, False, False]
    mask = np.asarray([bool(x) for x in components], dtype=bool)
    if mask.shape != (3,):
        raise ValueError(f"friction_ablation.components must be length 3, got {components}")

    model = raw_env._model
    names = names_from_raw(raw_env)
    peg_body = int(model.body(names.peg_body).id)
    tray_body = int(model.body(names.socket_body).id)
    peg_ids = _body_collision_geom_ids(model, peg_body)
    tray_ids = _body_collision_geom_ids(model, tray_body)

    def _scale(gids: list[int], scale: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if abs(scale - 1.0) < 1e-12:
            return rows
        for gid in gids:
            before = np.asarray(model.geom_friction[gid], dtype=np.float64).copy()
            after = before.copy()
            after[mask] = after[mask] * scale
            # Keep friction non-negative and avoid exact zeros (solver quirks).
            after = np.maximum(after, 1e-6)
            model.geom_friction[gid] = after
            gname = ""
            try:
                gname = str(model.geom(gid).name)
            except Exception:
                gname = f"geom_{gid}"
            rows.append(
                {
                    "gid": gid,
                    "name": gname,
                    "before": before.tolist(),
                    "after": after.tolist(),
                }
            )
        return rows

    peg_rows = _scale(peg_ids, peg_scale)
    tray_rows = _scale(tray_ids, tray_scale)
    meta = {
        "enabled": True,
        "privileged_diagnostic": True,
        "peg_body": str(names.peg_body),
        "tray_body": str(names.socket_body),
        "peg_scale": peg_scale,
        "tray_scale": tray_scale,
        "components_mask": mask.astype(int).tolist(),
        "peg_geoms_scaled": len(peg_rows),
        "tray_geoms_scaled": len(tray_rows),
        "peg_examples": peg_rows[:3],
        "tray_examples": tray_rows[:4],
    }
    print(
        "pci: friction_ablation "
        f"tray_scale={tray_scale:.3f} peg_scale={peg_scale:.3f} "
        f"tray_geoms={len(tray_rows)} peg_geoms={len(peg_rows)} "
        "(privileged_diagnostic)",
        flush=True,
    )
    return meta
