# utils/physics.py
import torch


def calc_eke(uv, clim):
    """Compute the eddy kinetic energy (EKE)."""
    eke = torch.sum((uv - clim.unsqueeze(0)) ** 2, dim=-2) / 2  # bs * n_points
    return eke


def iterate_input(t, input, num_var, predict, seq_l):
    input = input[:, :num_var * (seq_l - 1), ...]
    new_input = torch.cat([predict, input], dim=1)
    return t + 24., new_input