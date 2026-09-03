"""One-step constrained tip-tracking QP for wrist command.

Adapted concepts (cite-only runtime):
- Escande HQP: task-space tracking with bounds
- GraspQP bounded LS in priv_grasp_opt.solve_contact_forces_qp
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _planar(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - n * float(np.dot(v, n))


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit(normal)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = _unit(np.cross(a, n))
    t2 = np.cross(n, t1)
    return t1, t2


def _to2(v3: np.ndarray, normal: np.ndarray) -> np.ndarray:
    t1, t2 = _tangent_basis(normal)
    v = _planar(v3, normal)
    return np.array([float(np.dot(v, t1)), float(np.dot(v, t2))], dtype=np.float64)


def solve_wrist_qp(
    *,
    C: np.ndarray,
    err_tip3: np.ndarray,
    site: np.ndarray,
    normal: np.ndarray,
    max_step_m: float,
    lambda_reg: float = 0.12,
) -> np.ndarray:
    """Ridge LS: dw = (C'C + λI)^{-1} C' e, clip norm, map to 3D hold."""
    n = _unit(normal)
    t1, t2 = _tangent_basis(n)
    e2 = _to2(err_tip3, n)
    c = np.asarray(C, dtype=np.float64).reshape(2, 2)
    a = c.T @ c + lambda_reg * np.eye(2)
    b = c.T @ e2
    try:
        dw = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        dw = e2
    dn = float(np.linalg.norm(dw))
    if dn > max_step_m > 0.0:
        dw = dw * (max_step_m / dn)
    site3 = np.asarray(site, dtype=np.float64).reshape(3)
    return site3 + t1 * dw[0] + t2 * dw[1]
