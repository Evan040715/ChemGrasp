"""
Affordance definitions for Chem (and other) objects: no-grasp zones in object frame.

Object frame: origin at object center; Z axis = height (positive upward).
All coordinates in meters.
"""

import inspect
from typing import Union

import numpy as np
import torch

# Object short name: strip dataset prefix like "Chem+"
def _object_key(name: str) -> str:
    if "+" in name:
        return name.split("+", 1)[1]
    return name


# No-grasp rules per object (z in meters, relative to object center)
# Rule callable signatures supported:
# - rule(z): z is ndarray, shape (N,)
# - rule(z, y, x): z/y/x are ndarrays, shape (N,)
_AFFORDANCE_RULES = {
    # beaker_250ml: total height ~95mm, center at 47.5mm above base.
    # No-grasp: Z relative to center > 27.5mm (upper part near rim).
    # NOTE: uses elementwise boolean ops; `and` would be invalid for arrays.
    # NOTE: your edited condition `(x < -0.015) & (x > -0.015)` is impossible; we interpret it as a band
    #       `-0.015 < x < 0.015` to make the rule meaningful.
    "beaker_250ml": lambda z, y, x: (
        (z > 0.0375)
        | ((y < -0.032)
        & (x > -0.015)
        & (x < 0.015)
        & (z > 0.0275))
    ),

    # conical_beaker_250ml: total height ~100mm, center at 40mm above base.
    # No-grasp: Z > 50mm from center (upper); Z in [-40, -20] mm from center (bottom 20mm above base).
    "conical_beaker_250ml": lambda z: (z > 0.05) | ((z >= -0.04) & (z <= -0.02)),
    # Measuring_Cylinder: total height 315mm, center at 150mm above base.
    # No-grasp: Z relative to center > 150mm.
    "Measuring_Cylinder": lambda z: z > 0.15,
    # Round-bottom_Flask: total height 140mm, center at 55mm above base.
    # No-grasp: Z relative to center > 35mm.
    "Round-bottom_Flask": lambda z: z > 0.035,
}


def has_affordance(object_name: str) -> bool:
    """Return True if the object has a defined no-grasp affordance."""
    key = _object_key(object_name)
    return key in _AFFORDANCE_RULES


def is_no_grasp_zone(
    object_name: str,
    points: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute per-point no-grasp mask (True = point lies in no-grasp zone).

    :param object_name: str, e.g. 'beaker_250ml' or 'Chem+beaker_250ml'
    :param points: (N, 3) or (B, N, 3), xyz in object frame (meters); Z = height, origin at center.
    :return: boolean mask same shape as points[..., 0]; True = no-grasp (affordance zone).
    """
    key = _object_key(object_name)
    if key not in _AFFORDANCE_RULES:
        # No rule: return all False (no point is no-grasp)
        if isinstance(points, torch.Tensor):
            return torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
        return np.zeros(points.shape[:-1], dtype=bool)

    rule = _AFFORDANCE_RULES[key]
    if points.ndim == 3:
        B, N, _ = points.shape
        x = points[..., 0].reshape(-1)
        y = points[..., 1].reshape(-1)
        z = points[..., 2].reshape(-1)
    else:
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        B, N = None, None

    if isinstance(points, torch.Tensor):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        try:
            n_params = len(inspect.signature(rule).parameters)
        except (TypeError, ValueError):
            n_params = 1
        if n_params == 1:
            out_np = rule(z_np)
        elif n_params == 3:
            out_np = rule(z_np, y_np, x_np)
        else:
            raise TypeError(
                f"Affordance rule for '{key}' must accept 1 arg (z) or 3 args (z,y,x); got {n_params}."
            )
        out = torch.from_numpy(out_np).to(device=points.device, dtype=torch.bool)
    else:
        try:
            n_params = len(inspect.signature(rule).parameters)
        except (TypeError, ValueError):
            n_params = 1
        if n_params == 1:
            out = rule(z)
        elif n_params == 3:
            out = rule(z, y, x)
        else:
            raise TypeError(
                f"Affordance rule for '{key}' must accept 1 arg (z) or 3 args (z,y,x); got {n_params}."
            )

    if points.ndim == 3:
        out = out.reshape(B, N)
    return out
