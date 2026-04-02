from eulerian_fluid.utils import quaternion_to_rot_matrix, rotate_inertia
from eulerian_fluid.pcg_pressure_solve import project_standard

import torch
import torch.nn.functional as F


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_from_omega_dt(omega, dt, eps=1e-9):
    theta = torch.linalg.norm(omega) * dt
    half = 0.5 * theta
    w = torch.cos(half)
    s = torch.sin(half)
    axis = omega / (torch.linalg.norm(omega) + eps)
    return torch.stack([w, axis[0] * s, axis[1] * s, axis[2] * s], dim=-1)


def integrate_3d(u, rho, dt, gravity_accel=-9.8):
    gravity = torch.tensor([0.0, 0.0, gravity_accel], device=u.device, dtype=u.dtype)
    return u + gravity * dt * rho.unsqueeze(-1)


def make_base_grid(depth, height, width, device, dtype):
    z = torch.linspace(-1, 1, depth, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([x_grid, y_grid, z_grid], dim=-1)


def advect_3d(u0, rho0, dt, base_grid=None):
    depth, height, width, _ = u0.shape
    device, dtype = u0.device, u0.dtype

    if base_grid is None:
        base_grid = make_base_grid(depth, height, width, device, dtype)

    scale = torch.tensor(
        [2 / (width - 1), 2 / (height - 1), 2 / (depth - 1)],
        device=device,
        dtype=dtype,
    )
    u_xyz = torch.stack([u0[..., 2], u0[..., 1], u0[..., 0]], dim=-1)
    back_grid = base_grid - (u_xyz * dt) * scale

    u_in = u0.permute(3, 0, 1, 2).unsqueeze(0)
    rho_in = rho0.unsqueeze(0).unsqueeze(0)
    grid = back_grid.unsqueeze(0)

    u1 = F.grid_sample(u_in, grid, mode="bilinear", padding_mode="border", align_corners=True)
    rho1 = F.grid_sample(rho_in, grid, mode="bilinear", padding_mode="border", align_corners=True)

    u1 = u1.squeeze(0).permute(1, 2, 3, 0)
    rho1 = rho1.squeeze(0).squeeze(0)
    return u1, rho1


def rotate_and_shift_mask(fluid_mask, q, shift):
    solid_mask = 1.0 - fluid_mask

    depth, height, width = solid_mask.shape
    device = solid_mask.device
    dtype = solid_mask.dtype
    shift = shift.to(device=device, dtype=dtype)

    z = torch.arange(depth, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    z_grid, y_grid, x_grid = torch.meshgrid(z, y, x, indexing="ij")
    pos = torch.stack([z_grid, y_grid, x_grid], dim=-1)

    weight_sum = solid_mask.sum().clamp_min(1e-8)
    pivot = (pos * solid_mask.unsqueeze(-1)).sum(dim=(0, 1, 2)) / weight_sum

    rotation = quaternion_to_rot_matrix(q).to(device=device, dtype=dtype)
    back = torch.einsum("ij,dhwj->dhwi", rotation.T, (pos - shift - pivot)) + pivot

    bz = back[..., 0].clamp(0.0, depth - 1.001)
    by = back[..., 1].clamp(0.0, height - 1.001)
    bx = back[..., 2].clamp(0.0, width - 1.001)

    lz = bz.floor().long()
    ly = by.floor().long()
    lx = bx.floor().long()
    tz = bz - lz.to(dtype)
    ty = by - ly.to(dtype)
    tx = bx - lx.to(dtype)

    def trilerp(values):
        c000 = values[lz, ly, lx]
        c100 = values[lz + 1, ly, lx]
        c010 = values[lz, ly + 1, lx]
        c001 = values[lz, ly, lx + 1]
        c110 = values[lz + 1, ly + 1, lx]
        c101 = values[lz + 1, ly, lx + 1]
        c011 = values[lz, ly + 1, lx + 1]
        c111 = values[lz + 1, ly + 1, lx + 1]

        c00 = c000 * (1 - tz) + c100 * tz
        c01 = c001 * (1 - tz) + c101 * tz
        c10 = c010 * (1 - tz) + c110 * tz
        c11 = c011 * (1 - tz) + c111 * tz
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tx) + c1 * tx

    return 1.0 - trilerp(solid_mask)


def make_grid_centers(nx, ny, nz, dx, device):
    xs = (torch.arange(nx, device=device) + 0.5) * dx
    ys = (torch.arange(ny, device=device) + 0.5) * dx
    zs = (torch.arange(nz, device=device) + 0.5) * dx
    x_grid, y_grid, z_grid = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([x_grid, y_grid, z_grid], dim=-1)


def rigid_velocity_field(grid_xyz, x_cm, v_cm, omega):
    offset = grid_xyz - x_cm.view(1, 1, 1, 3)
    omega_grid = omega.view(1, 1, 1, 3)
    v_grid = v_cm.view(1, 1, 1, 3)
    return v_grid + torch.cross(omega_grid.expand_as(offset), offset, dim=-1)


def brinkman_step(u, fluid_mask, solid_pos, solid_v, solid_omega, dt, lam, rho0=1.0, dx=1.0):
    chi = (1.0 - fluid_mask).unsqueeze(-1)

    nx, ny, nz, _ = u.shape
    grid_xyz = make_grid_centers(nx, ny, nz, dx, u.device)
    u_s = rigid_velocity_field(grid_xyz, solid_pos, solid_v, solid_omega)

    denom = 1.0 + dt * lam * chi
    u_new = (u + dt * lam * chi * u_s) / denom

    du = u_new - u
    f_on_fluid = rho0 * du / dt
    cell_vol = dx ** 3

    force_on_body = -f_on_fluid.sum(dim=(0, 1, 2)) * cell_vol
    offset = grid_xyz - solid_pos.view(1, 1, 1, 3)
    torque_on_body = -(torch.cross(offset, f_on_fluid, dim=-1).sum(dim=(0, 1, 2)) * cell_vol)

    return u_new, force_on_body, torque_on_body


def implicit_brinkman_omega_update(u, fluid_mask, solid_pos, solid_v, I_world, L_world, dt, lam, rho0=1.0, dx=1.0):
    chi = 1.0 - fluid_mask
    nx, ny, nz, _ = u.shape
    grid_xyz = make_grid_centers(nx, ny, nz, dx, u.device)
    offset = grid_xyz - solid_pos.view(1, 1, 1, 3)

    alpha = (rho0 * lam * chi) / (1.0 + dt * lam * chi)
    alpha_grid = alpha.unsqueeze(-1)
    cell_vol = dx ** 3

    v_minus_u = solid_v.view(1, 1, 1, 3) - u
    tau0 = -(torch.cross(offset, alpha_grid * v_minus_u, dim=-1).sum(dim=(0, 1, 2)) * cell_vol)

    rr_t = offset.unsqueeze(-1) * offset.unsqueeze(-2)
    offset_sq = (offset * offset).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
    eye3 = torch.eye(3, device=u.device).view(1, 1, 1, 3, 3)
    cell_term = rr_t - offset_sq * eye3
    K = (alpha.unsqueeze(-1).unsqueeze(-1) * cell_term).sum(dim=(0, 1, 2)) * cell_vol

    A = I_world - dt * K + 1e-9 * torch.eye(3, device=u.device)
    b = L_world + dt * tau0
    omega_new = torch.linalg.solve(A, b)

    L_new = I_world @ omega_new
    return omega_new, L_new, tau0, K


def sim_step(
    u_t,
    rho_t,
    p_t,
    fluid_mask,
    solid_inertia_body,
    solid_mass,
    solid_pos,
    solid_v,
    solid_q,
    solid_L_world,
    num_pressure_iterations,
    dt,
    before_advect=None,
    move_object=False,
    dx=1.0,
    brinkman_lambda=100.0,
):
    fluid_mask = rotate_and_shift_mask(fluid_mask, solid_q, solid_pos)
    u_t = integrate_3d(u_t, rho_t, dt)

    I_world = rotate_inertia(solid_inertia_body, solid_q)
    u_before_brink = u_t.clone()
    do_rotate = True

    if do_rotate:
        solid_omega, solid_L_world, _, _ = implicit_brinkman_omega_update(
            u_before_brink,
            fluid_mask,
            solid_pos,
            solid_v,
            I_world,
            solid_L_world,
            dt,
            brinkman_lambda,
            rho0=1.0,
            dx=dx,
        )
    else:
        solid_omega = torch.zeros(3, device=u_t.device, dtype=u_t.dtype)

    u_t, force_on_body, _ = brinkman_step(
        u_t,
        fluid_mask,
        solid_pos,
        solid_v,
        solid_omega,
        dt,
        brinkman_lambda,
        rho0=1.0,
        dx=dx,
    )

    if before_advect is None:
        before_advect = lambda u, rho: (u, rho)
    u_t, rho_t = before_advect(u_t, rho_t)

    u_t, rho_t = advect_3d(u_t, rho_t, dt)
    rho_t = rho_t.clamp_min(0.0)
    u_t, p_t = project_standard(u_t, iters=num_pressure_iterations, dx=dx)

    if move_object:
        mass = solid_mass.detach()
        force = force_on_body
        force = force + torch.tensor([0.0, 0.0, -0.98], device=solid_v.device, dtype=solid_v.dtype) * mass

        thrust_dir = torch.tensor([0.0, -1.0, 0.0], device=solid_v.device, dtype=solid_v.dtype)
        thrust_dir = thrust_dir / (thrust_dir.norm() + 1e-9)
        force = force + thrust_dir * (17.4 * mass)

        solid_v = solid_v + (force / mass) * dt
        solid_pos = solid_pos + solid_v * dt

        if do_rotate:
            I_world_new = rotate_inertia(solid_inertia_body, solid_q)
            solid_omega = torch.linalg.solve(
                I_world_new + 1e-9 * torch.eye(3, device=solid_L_world.device),
                solid_L_world,
            )
            dq = quat_from_omega_dt(solid_omega, dt)
            solid_q = quaternion_multiply(dq, solid_q)
            solid_q = solid_q / (solid_q.norm() + 1e-9)
    else:
        solid_v = torch.zeros(3, device=u_t.device, dtype=u_t.dtype)

    return u_t, rho_t, p_t, solid_pos, solid_v, solid_q, solid_L_world
