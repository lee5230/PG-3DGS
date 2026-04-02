
"""
Loss registry and composable loss functions for fluid/solid simulation experiments.

This module provides:
- A registry for loss functions, allowing easy composition and experimentation.
- `compute_total_loss`, which computes a weighted sum of registered losses given a list of specs and simulation state.
- Several example loss functions, each documented below.

Loss specification format:
    loss_specs = [
        {"name": "maximize_z_position", "weight": 0.5, "args": {}},
    ]

Each loss function receives a config dict (`cfg`, from `args`) and a simulation state dict (`state`).
The state dict typically contains: `rho_list`, `pressure_list`, `solid_list`, `solid_pos_list`, `solid_q_list`, `device`.
All loss functions return a scalar torch.Tensor.
"""

import torch

_LOSS_REGISTRY = {}

def register_loss(name):
    """Decorator to register a loss function under a given name."""
    def _decorator(fn):
        _LOSS_REGISTRY[name] = fn
        return fn
    return _decorator

def compute_total_loss(loss_specs, state):
    """
    Computes the weighted sum of registered losses.

    Args:
        loss_specs: list of dicts like [{'name': ..., 'weight': ..., 'args': {...}}, ...]
        state: dict containing simulation outputs needed by loss functions.
    Returns:
        total_loss_tensor: torch.Tensor
        breakdown_dict: dict mapping loss name to float value
    """
    total = torch.tensor(0.0, device=state.get('device', torch.device('cpu')))
    breakdown = {}
    for spec in loss_specs:
        name = spec['name']
        weight = spec.get('weight', 1.0)
        args = spec.get('args', {})
        fn = _LOSS_REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown loss '{name}'")
        val = fn(args, state)
        val = val * weight
        total = total + val
        breakdown[name] = val.detach().cpu().item() if isinstance(val, torch.Tensor) else float(val)
    return total, breakdown



@register_loss('weighted_density')
def loss_weighted_density(cfg, state):
    """
    Computes a weighted density loss for the final density field in a simulation.
    The loss is calculated by multiplying the final density (`rho_t`) by a set of linearly spaced weights along the last dimension,
    then taking the mean and returning its negative value. This encourages higher density values towards the end of the weighted dimension.
    Args:
        cfg: Configuration object (unused in this function, but included for interface consistency).
        state (dict): Dictionary containing simulation state. Must include:
            - 'rho_list': List of density tensors, where the last element is used.
            - 'device': The torch device for tensor operations.
    Returns:
        torch.Tensor: Scalar tensor representing the negative mean of the weighted density.
                      Returns 0.0 if the final density is None.
    """
    rho_t = state['rho_list'][-1]  # final rho
    if rho_t is None:
        return torch.tensor(0.0, device=state.get('device'))
    W = rho_t.shape[2]
    z_weights = torch.linspace(0, 1, W, device=state.get('device'))
    z_weights = torch.exp(z_weights) * z_weights

    # get max density value at each z level
    max_density_per_z = torch.amax(rho_t, dim=(0, 1))  # shape (W,)
    # try logsumexp for stability
    # You can control the "sharpness" of logsumexp by scaling the input with a temperature parameter.
    # Lower temperature -> sharper (closer to max), higher temperature -> smoother (closer to mean).
    temperature = 0.1  # Add this to your args/config
    # max_density_per_z = torch.logsumexp(rho_t / temperature, dim=(0, 1)) * temperature 
    weighted_density = max_density_per_z * z_weights
    # weighted_density = rho_t * z_weights
    return -weighted_density.mean()

@register_loss('density_at_bottom')
def loss_density_at_bottom(cfg, state):
    """
    Penalizes lack of density in the bottom third of the domain.
    Args:
        cfg: unused
        state: expects 'rho_list'
    Returns:
        Negative mean density in bottom third of z axis in final frame.
    """
    rho_t = state['rho_list'][-1]  # final rho
    # soft mask representing solid locations
    # solid_mask = state.get('solid_list', [None])[-1]
    # # get lowest solid z index greater than 0.5
    # if solid_mask is not None:
    #     solid_z_indices = torch.where(solid_mask.max(dim=(0, 1))[0] > 0.5)[0]
    #     if len(solid_z_indices) > 0:
    #         min_solid_z = solid_z_indices.min().item()
    W = rho_t.shape[2]

    return -torch.mean(rho_t[:, :, :W//3])


@register_loss('maximize_z_position')
def loss_maximize_z_position(cfg, state):
    """
    Rewards higher average Z position of the solid over time (negative loss).
    Args:
        cfg: unused
        state: expects 'solid_pos_list', 'device'
    Returns:
        Negative mean of average z position across all timesteps.
    """
    solid_pos_list = state['solid_pos_list']
    loss = torch.tensor(0.0, device=state.get('device'))
    for solid_pos_t in solid_pos_list:
        loss = loss - solid_pos_t[2].mean()  # maximize average z position
    return loss / len(solid_pos_list)


@register_loss('maintain_x_position')
def loss_maintain_x_position(cfg, state):
    """
    Penalizes changes in the object's mean X position relative to the start.
    Args:
        cfg: unused
        state: expects 'solid_pos_list', 'device'
    Returns:
        Mean absolute change in x position across all timesteps.
    """
    solid_pos_list = state['solid_pos_list']
    orig_x = solid_pos_list[0][0]
    loss = torch.tensor(0.0, device=state.get('device'))
    for solid_pos_t in solid_pos_list:
        loss = loss + (solid_pos_t[0].mean() - orig_x.mean()).abs()  # minimize change in average x position
    return loss / len(solid_pos_list)

