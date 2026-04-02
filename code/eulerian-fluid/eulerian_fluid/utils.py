import torch
import torch.nn.functional as F


def quaternion_to_rot_matrix(q):
    w, x, y, z = q.unbind(-1)

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotate_inertia(inertia_body, q):
    rotation = quaternion_to_rot_matrix(q)
    return rotation @ inertia_body @ rotation.transpose(-1, -2)


def inertia_from_chi(chi, dx=1.0, rho_s=1.0):
    device = chi.device
    dtype = chi.dtype
    nx, ny, nz = chi.shape
    cell_vol = dx ** 3

    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * dx
    ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * dx
    zs = (torch.arange(nz, device=device, dtype=dtype) + 0.5) * dx
    x_grid, y_grid, z_grid = torch.meshgrid(xs, ys, zs, indexing="ij")
    pos = torch.stack([x_grid, y_grid, z_grid], dim=-1)

    mass_density = rho_s * chi * cell_vol
    total_mass = mass_density.sum() + 1e-12
    center = (mass_density.unsqueeze(-1) * pos).sum(dim=(0, 1, 2)) / total_mass

    offset = pos - center.view(1, 1, 1, 3)
    offset_sq = (offset * offset).sum(dim=-1)

    inertia = torch.eye(3, device=device, dtype=dtype) * (mass_density * offset_sq).sum()
    inertia -= torch.einsum("ijk,ijkl,ijkm->lm", mass_density, offset, offset)
    return total_mass, center, inertia


def prune_disconnected(
    mask,
    low_tau=0.2,
    high_tau=0.6,
    return_soft=True,
    max_iters=1024,
):
    if mask.ndim != 3:
        raise ValueError("mask must have shape [D, H, W]")

    depth, height, width = mask.shape
    x = mask[None, None]

    support = (x >= low_tau).to(x.dtype)
    seeds = (x >= high_tau).to(x.dtype) * support
    if seeds.max() == 0:
        if support.max() == 0:
            return torch.zeros_like(mask)
        seeds = torch.zeros_like(x)
        seeds.view(-1)[torch.argmax(x * support)] = 1.0

    def pool3(tensor):
        return F.max_pool3d(tensor, kernel_size=3, stride=1, padding=1)

    grown = seeds
    for _ in range(max_iters):
        next_grown = (pool3(grown) * support).clamp(max=1.0)
        if torch.equal(next_grown > 0.5, grown > 0.5):
            grown = next_grown
            break
        grown = next_grown

    foreground = (grown > 0.5).squeeze(0).squeeze(0)
    if foreground.max() == 0:
        return torch.zeros_like(mask)

    lin_ids = torch.arange(
        1,
        depth * height * width + 1,
        device=mask.device,
        dtype=torch.float64,
    ).view(depth, height, width)
    labels = torch.where(foreground, lin_ids, torch.zeros((), device=mask.device, dtype=torch.float64))
    labels = labels[None, None]

    propagated = labels
    foreground = foreground[None, None]
    for _ in range(max_iters):
        next_labels = torch.where(foreground, pool3(propagated), torch.zeros_like(propagated))
        if torch.allclose(next_labels, propagated):
            propagated = next_labels
            break
        propagated = next_labels

    final_labels = propagated.squeeze(0).squeeze(0).to(torch.long).view(-1)
    max_id = int(final_labels.max().item())
    counts = torch.bincount(final_labels, minlength=max_id + 1)
    counts[0] = 0
    largest_id = int(torch.argmax(counts).item())
    largest_component = (final_labels == largest_id).view(depth, height, width).to(mask.dtype)

    if return_soft:
        return mask * largest_component
    return largest_component
