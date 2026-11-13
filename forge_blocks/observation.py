"""Observation maths that differs from upstream ForgeEnv. Needs only torch, so it can be tested
without Isaac Sim. Quaternions are (w, x, y, z).
"""

import math

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rotation vector for q, taking the short way round."""
    q = q * torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    mag = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    half_angle = torch.atan2(mag, q[..., :1])
    scale = torch.where(mag > eps, 2.0 * half_angle / mag.clamp_min(eps), 2.0 * torch.ones_like(mag))
    return q[..., 1:] * scale


def noisy_finite_difference(noisy_pos, prev_noisy_pos, noisy_quat, prev_noisy_quat, dt):
    """Velocities from consecutive noisy samples, as the FORGE paper describes (App. A).

    Returns (linvel, angvel). Angular velocity keeps only z, like ForgeEnv.
    """
    linvel = (noisy_pos - prev_noisy_pos) / dt
    angvel = axis_angle_from_quat(quat_mul(noisy_quat, quat_conjugate(prev_noisy_quat))) / dt
    angvel[..., 0:2] = 0.0
    return linvel, angvel


def sample_radius_error(n: int, radius_range, mode: str, device="cpu") -> torch.Tensor:
    """Position errors of shape (n, 3) with radius uniform in radius_range [m].

    "xyz" picks a uniform direction on the sphere (the paper's evaluation, Sec. V-B).
    "xy" picks a uniform direction in the plane and leaves z at zero.
    """
    r_min, r_max = radius_range
    radius = r_min + (r_max - r_min) * torch.rand(n, 1, device=device)
    if mode == "xyz":
        direction = torch.randn(n, 3, device=device)
        return radius * direction / torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1e-9)
    if mode == "xy":
        theta = 2 * math.pi * torch.rand(n, device=device)
        direction = torch.stack((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=1)
        return radius * direction
    raise ValueError(f"unknown error mode {mode!r}; expected 'xy' or 'xyz'")


def force_deadband(force: torch.Tensor, deadband: float, mode: str = "axis") -> torch.Tensor:
    """Shrink the observed force toward zero by `deadband` N; anything smaller reads 0.

    "axis": per component (a 4 N sideways push vanishes under a 5 N band).
    "norm": on the magnitude, keeping the direction of what is left.
    """
    if mode == "axis":
        return torch.sign(force) * (force.abs() - deadband).clamp_min(0.0)
    if mode == "norm":
        mag = torch.linalg.norm(force, dim=-1, keepdim=True)
        return force * (mag - deadband).clamp_min(0.0) / mag.clamp_min(1e-9)
    raise ValueError(f"unknown dead-band mode {mode!r}; expected 'axis' or 'norm'")
