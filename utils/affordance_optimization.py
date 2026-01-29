"""
Affordance penalty terms for IK optimization (CVXPY).

Maps structured no-grasp zone descriptors to CVXPY penalty expressions
so that create_problem can add them to the objective without embedding
affordance logic inside optimization.py.

Object frame: origin at object center; Z = height (positive upward).
All coordinates in meters.
"""

import cvxpy as cp
from typing import List, Any


def _object_key(name: str) -> str:
    """Strip dataset prefix like 'Chem+'."""
    if "+" in name:
        return name.split("+", 1)[1]
    return name


# Structured descriptors per object (no-grasp zones).
# Each descriptor is a dict; build_affordance_penalty(p, desc) returns a CVXPY scalar.
# Types: "z_above" (DPP-safe), "z_band", "compound_and" (z_band/compound_and can break DPP in cvxpylayers).
# We use only z_above here so CvxpyLayer accepts the problem; z_band/compound_and kept for future DPP-safe formulation.
_AFFORDANCE_DESCRIPTORS: dict = {
    "Round-bottom_Flask": [
        {"type": "z_above", "z_max": 0.035},
    ],
    "Measuring_Cylinder": [
        {"type": "z_above", "z_max": 0.15},
    ],
    "conical_beaker_250ml": [
        {"type": "z_above", "z_max": 0.05},
        # z_band (forbidden z in [-0.04, -0.02]) breaks DPP; omit here, use post-hoc filter or DPP-safe reformulation later
    ],
    "beaker_250ml": [
        {"type": "z_above", "z_max": 0.0375},
        {"type": "z_above", "z_max": 0.0275},
    ],
}


def get_affordance_descriptors(object_name: str) -> List[dict]:
    """
    Return the list of no-grasp zone descriptors for the given object.
    Empty list if the object has no affordance for optimization.
    """
    key = _object_key(object_name)
    return _AFFORDANCE_DESCRIPTORS.get(key, [])


def _violation_expr(p: Any, term: dict) -> Any:
    """
    Return a CVXPY expression that is positive when the point p (3-vector)
    lies in the forbidden sub-region described by term.
    p is cvxpy expression with p[0], p[1], p[2] = x, y, z.
    """
    t = term["type"]
    ax = term.get("axis", 2)
    if t == "above":
        # forbidden when p[ax] > val  ->  violation = p[ax] - val
        return p[ax] - term["val"]
    if t == "below":
        # forbidden when p[ax] < val  ->  violation = val - p[ax]
        return term["val"] - p[ax]
    if t == "band":
        # forbidden when low <= p[ax] <= high  ->  violation = min(p[ax]-low, high-p[ax])
        return cp.minimum(p[ax] - term["low"], term["high"] - p[ax])
    raise ValueError(f"Unknown term type: {t}")


def build_affordance_penalty(p: Any, desc: dict) -> Any:
    """
    Build CVXPY penalty expression for one descriptor and one link position p.

    :param p: CVXPY expression of shape (3,) = (x, y, z) in object frame (e.g. predict_frame_xyz).
    :param desc: One descriptor dict from get_affordance_descriptors().
    :return: Scalar CVXPY expression (non-negative); zero when p is outside the forbidden zone.
    """
    t = desc["type"]
    if t == "z_above":
        # penalty = max(0, p[2] - z_max)^2  (single scalar, use sum_squares for DCP)
        return cp.sum_squares(cp.maximum(0, p[2] - desc["z_max"]))
    if t == "z_band":
        # penalty when z in [z_low, z_high]: max(0, min(p[2]-z_low, z_high-p[2]))^2
        return cp.sum_squares(
            cp.maximum(0, cp.minimum(p[2] - desc["z_low"], desc["z_high"] - p[2]))
        )
    if t == "compound_and":
        # penalty when all terms are violated: max(0, min(v1, v2, ...))^2
        terms = desc["terms"]
        v0 = _violation_expr(p, terms[0])
        for term in terms[1:]:
            v0 = cp.minimum(v0, _violation_expr(p, term))
        return cp.sum_squares(cp.maximum(0, v0))
    raise ValueError(f"Unknown descriptor type: {t}")
