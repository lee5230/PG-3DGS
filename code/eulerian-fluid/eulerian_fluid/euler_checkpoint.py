import math
import torch
import torch.utils.checkpoint as cp
from eulerian_fluid.fluid_visualization import visualize_3d_fluid_animation_vispy, visualize_solid_mask_slice, visualize_solid_mask_cutaway_mesh_pyvista, pyvista_is_available
from eulerian_fluid.fluid_logic_3d import sim_step, rotate_and_shift_mask
from eulerian_fluid.utils import inertia_from_chi
import json
import matplotlib.pylab as plt
from skimage import measure
import trimesh
from eulerian_fluid.losses import compute_total_loss

from eulerian_fluid.utils import prune_disconnected, rotate_inertia

import os

from gaussian_splatting.gauss_render import build_scaling_rotation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_visualization_mask_from_source(
    *,
    grid_shape,
    resolution_scale: float = 2.0,
    extent_scale: float = 1.4,
    voxels_per_unit: float = 32.0,
    mask_sharpening: float = 0.05,
    tile_size: int = 8,
    gaussian_means=None,
    gaussian_scalings=None,
    gaussian_rotations=None,
    gaussian_opacities=None,
):
    """
    Rebuild a visualization mask directly from the source representation on a
    larger grid, keeping `voxels_per_unit` fixed.

    The grid is first increased by `resolution_scale`, then enlarged again by
    `extent_scale` to add extra margin around the object.
    """
    if grid_shape is None:
        raise ValueError("grid_shape must be provided to rebuild the visualization mask.")

    visualization_grid_shape = tuple(
        max(int(math.ceil(dim * float(resolution_scale) * float(extent_scale))), int(dim))
        for dim in grid_shape
    )

    missing_gaussian_inputs = [
        name
        for name, value in (
            ('gaussian_means', gaussian_means),
            ('gaussian_scalings', gaussian_scalings),
            ('gaussian_rotations', gaussian_rotations),
            ('gaussian_opacities', gaussian_opacities),
        )
        if value is None
    ]
    if missing_gaussian_inputs:
        raise ValueError(
            'Missing Gaussian inputs for visualization mask rebuild: '
            + ', '.join(missing_gaussian_inputs)
        )

    viz_mask = gaussians_to_fluid_mask(
        gaussian_means,
        gaussian_scalings,
        gaussian_rotations,
        gaussian_opacities,
        grid_shape=visualization_grid_shape,
        voxels_per_unit=voxels_per_unit,
        tile_size=tile_size,
    )

    return viz_mask ** mask_sharpening, visualization_grid_shape


def build_soft_mask_tiled(
    centers,
    covariances,
    opacities,
    grid_shape,
    radius_scale=3.0,
    tile_size=8,
):
    D, H, W = grid_shape
    device = centers.device
    voxel_var = 1.0 / 12.0

    min_sigma = 0.5
    I3 = torch.eye(3, device=device, dtype=covariances.dtype).unsqueeze(0)
    eps_inv = 1e-6

    with torch.no_grad():
        cov_eff = covariances + voxel_var * I3 + (min_sigma ** 2) * I3
        half_extents = radius_scale * torch.sqrt(
            torch.diagonal(cov_eff, dim1=-2, dim2=-1).clamp_min(0.0)
        )

        grid_dims = torch.tensor([D, H, W], device=device)
        x_min = centers - half_extents
        x_max = centers + half_extents

        box_min_all = torch.floor(x_min).clamp(min=0).long()
        box_max_all = torch.ceil(x_max).clamp(max=grid_dims).long()

    final_mask = torch.zeros((D, H, W), device=device)

    for z_start in range(0, D, tile_size):
        for y_start in range(0, H, tile_size):
            for x_start in range(0, W, tile_size):
                z_end = min(z_start + tile_size, D)
                y_end = min(y_start + tile_size, H)
                x_end = min(x_start + tile_size, W)

                tile_min = torch.tensor([z_start, y_start, x_start], device=device)
                tile_max = torch.tensor([z_end, y_end, x_end], device=device)

                overlap_min = torch.maximum(box_min_all, tile_min)
                overlap_max = torch.minimum(box_max_all, tile_max)
                gaussians_in_tile_mask = torch.all(overlap_min < overlap_max, dim=1)

                if not gaussians_in_tile_mask.any():
                    continue

                centers_tile = centers[gaussians_in_tile_mask]
                covs_tile = covariances[gaussians_in_tile_mask]
                opacities_tile = opacities[gaussians_in_tile_mask]

                z_range = torch.arange(z_start, z_end, device=device, dtype=torch.float32) + 0.5
                y_range = torch.arange(y_start, y_end, device=device, dtype=torch.float32) + 0.5
                x_range = torch.arange(x_start, x_end, device=device, dtype=torch.float32) + 0.5

                if z_range.numel() == 0 or y_range.numel() == 0 or x_range.numel() == 0:
                    continue

                Z, Y, X = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
                local_grid_flat = torch.stack([Z, Y, X], dim=-1).view(-1, 3).float()
                deltas = local_grid_flat.unsqueeze(1) - centers_tile.unsqueeze(0)
                I = I3[:covs_tile.shape[0]]
                cov = covs_tile
                cov_eff = cov + voxel_var * I + min_sigma ** 2 * I
                cov_eff_safe = cov_eff + eps_inv * I

                cov_inv = torch.linalg.inv(cov_eff_safe)
                dM2 = torch.einsum('vpi,pij,vpj->vp', deltas, cov_inv, deltas)
                weight = torch.exp(-0.5 * dM2)

                alphas = (opacities_tile.view(1, -1) * weight).clamp(max=0.99)
                occupancy = 1.0 - torch.prod(1.0 - alphas, dim=1)
                tile_mask = occupancy.reshape(z_end - z_start, y_end - y_start, x_end - x_start)
                final_mask[z_start:z_end, y_start:y_end, x_start:x_end] = tile_mask

    return final_mask


def compute_opacity_weighted_center(means, opacities, weight_floor=1e-3, total_weight_eps=1e-8):
    """
    Compute an opacity-weighted Gaussian center. Falls back to the uniform mean when
    the raw opacity mass is effectively zero.
    """
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"Expected means with shape (N, 3), got {tuple(means.shape)}")

    raw_weights = opacities.reshape(-1).to(device=means.device, dtype=means.dtype)
    if raw_weights.numel() != means.shape[0]:
        raise ValueError(
            f"Opacity count {raw_weights.numel()} does not match number of means {means.shape[0]}"
        )

    if raw_weights.sum().detach().item() <= total_weight_eps:
        return means.mean(dim=0)

    weights = raw_weights.clamp_min(weight_floor)
    return (means * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()


def normalize_gaussians_fixed_scale(means, covariances, opacities, grid_shape, voxels_per_unit):
    """
    Convert Gaussian-space coordinates into voxel-space coordinates with a fixed scale.
    The object is re-centered by opacity-weighted COM on every call.
    """
    D, H, W = grid_shape
    center = compute_opacity_weighted_center(means, opacities)
    scale = torch.as_tensor(voxels_per_unit, device=means.device, dtype=means.dtype)
    grid_center = torch.tensor([D, H, W], device=means.device, dtype=means.dtype) / 2

    new_means = (means - center) * scale + grid_center
    new_covariances = covariances * (scale ** 2)

    return new_means, new_covariances


def gaussians_to_fluid_mask(
    means,
    scaling,
    rotation,
    opacities,
    grid_shape,
    voxels_per_unit=32.0,
    tile_size=8,
):
    D, H, W = grid_shape

    L = build_scaling_rotation(scaling, rotation)
    covariances = L @ L.transpose(1, 2)

    means, covariances = normalize_gaussians_fixed_scale(
        means,
        covariances,
        opacities,
        (D, H, W),
        voxels_per_unit=voxels_per_unit,
    )

    return 1 - build_soft_mask_tiled(
        means,
        covariances,
        opacities,
        grid_shape=(D, H, W),
        radius_scale=5.0,
        tile_size=tile_size
    )


def sim_block(u_t, rho_t, p_t, solid_mask, solid_inertia, solid_mass, solid_pos, solid_v, solid_q, solid_L_world, block_size, dt, num_pressure_iterations, save_intermediate=False, move_object=False, brinkman_lambda=1e4):
    intermediates = {
        'u': [],
        'rho': [],
        'solid_pos': [],
        'solid_q': [],
        'p': [],
    }
    for _ in range(block_size):
        u_t, rho_t, p_t, solid_pos, solid_v, solid_q, solid_L_world = sim_step(u_t, rho_t, p_t, solid_mask, solid_inertia, solid_mass, solid_pos, solid_v, solid_q, solid_L_world, num_pressure_iterations, dt, before_advect=None, move_object=move_object, brinkman_lambda=brinkman_lambda)
        if save_intermediate:
            intermediates['u'].append(u_t.detach().cpu())
            intermediates['rho'].append(rho_t.detach().cpu())
            intermediates['solid_pos'].append(solid_pos.detach().cpu())
            intermediates['solid_q'].append(solid_q.detach().cpu())
            intermediates['p'].append(p_t.detach().cpu())
    if save_intermediate:
        return u_t, rho_t, p_t, solid_pos, solid_v, solid_q, solid_L_world, intermediates
    else:
        return u_t, rho_t, p_t, solid_pos, solid_v, solid_q, solid_L_world


def init_state(grid_size=None, solid_mask=None, blob_at_com=False, blob_radius=5.5):
    D, H, W = grid_size
    u = torch.zeros((D, H, W, 3), device=device)
    rho = torch.zeros((D, H, W), device=device)

    if blob_at_com:
        if solid_mask is None:
            raise ValueError("solid_mask must be provided if blob_at_com is True")
        indices = torch.nonzero(solid_mask < 0.5, as_tuple=False)
        if indices.shape[0] == 0:
            com = torch.tensor([D / 2, H / 2, W / 2], device=device)
        else:
            com = indices.float().mean(dim=0)
        object_com = com.tolist()

        Z, Y, X = torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((Z - object_com[0])**2 + (Y - object_com[1])**2 + (X - object_com[2])**2)
        mask = dist < blob_radius
        rho[mask] = 1.0

    return u, rho


def simulate_physics_loss(
    gaussian_means=None,
    gaussian_scalings=None,
    gaussian_rotations=None,
    gaussian_opacities=None,
    grid_shape=None,
    num_steps=30,
    num_pressure_iterations=10,
    dt=0.15,
    step=0,
    save_dir='.',
    disable_checkpoint=False,
    checkpoint_interval=10,
    tile_size=8,
    voxels_per_unit=32.0,
    do_visualize=True,
    mask_sharpening=0.05,
    init_solid_pos=(0, 0, 0),
    init_solid_q=(1, 0, 0, 0),
    init_solid_v=(0, 0, 0),
    init_solid_omega=(0, 0, 0),
    loss_specs=None,
    move_object=False,
    com_blob_radius=0.0,
    brinkman_lambda=1e4,
):
    if grid_shape is None:
        raise ValueError("grid_shape must be provided")

    D, H, W = grid_shape

    rho = []
    solid_list = []
    u_list = []
    solid_pos_list = []
    solid_q_list = []
    pressure_list = []

    p_visualize = []
    solid_visualize = []
    u_visualize = []
    rho_visualize = []
    solid_q_visualize = []
    solid_pos_visualize = []
    
    missing_gaussian_inputs = [
        name
        for name, value in (
            ('gaussian_means', gaussian_means),
            ('gaussian_scalings', gaussian_scalings),
            ('gaussian_rotations', gaussian_rotations),
            ('gaussian_opacities', gaussian_opacities),
        )
        if value is None
    ]
    if missing_gaussian_inputs:
        raise ValueError(
            'Missing Gaussian inputs for gaussian-based physics mask: '
            + ', '.join(missing_gaussian_inputs)
        )

    fluid_mask = gaussians_to_fluid_mask(
        gaussian_means,
        gaussian_scalings,
        gaussian_rotations,
        gaussian_opacities,
        grid_shape=grid_shape,
        voxels_per_unit=voxels_per_unit,
        tile_size=tile_size,
    )

    fluid_mask = fluid_mask ** mask_sharpening

    surface = 0.5

    if torch.isnan(fluid_mask).any():
        raise ValueError("NaN detected in fluid_mask")
    if torch.isinf(fluid_mask).any():
        raise ValueError("Infinite value detected in fluid_mask")

    solid_v = torch.tensor(init_solid_v, device=device)
    solid_pos = torch.tensor(init_solid_pos, device=device)
    solid_q = torch.tensor(init_solid_q, device=device)
    solid_omega = torch.tensor(init_solid_omega, device=device)

    solid_mass, solid_com, solid_inertia = inertia_from_chi(1.0 - fluid_mask, dx=1.0, rho_s=7.5)
    I_world = rotate_inertia(solid_inertia, solid_q)
    solid_L_world = I_world @ solid_omega

    original_pos = solid_pos.clone()
    initial_q = solid_q.clone()

    translated_fluid_mask = rotate_and_shift_mask(fluid_mask, solid_q, solid_pos)

    u_t, rho_t = init_state(grid_size=grid_shape, solid_mask=translated_fluid_mask, blob_at_com=True, blob_radius=com_blob_radius)

    p_t = torch.zeros((D, H, W), device=device)

    for t in range(0, num_steps, checkpoint_interval):
        sim_block_args = (u_t, rho_t, p_t, fluid_mask, solid_inertia, solid_mass, solid_pos, solid_v, solid_q, solid_L_world, checkpoint_interval, dt, num_pressure_iterations, True, move_object, brinkman_lambda)

        if not disable_checkpoint:
            result = cp.checkpoint(
                sim_block, *sim_block_args, use_reentrant=True
            )
        else:
            result = sim_block(*sim_block_args)

        u_t, rho_t, p_t, solid_pos, solid_v, solid_q, solid_L_world, intermediates = result

        def check_tensor(tensor, name):
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN detected in tensor: {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Infinite value detected in tensor: {name}")

        check_tensor(u_t, "u_t")
        check_tensor(rho_t, "rho_t")
        check_tensor(p_t, "p_t")
        check_tensor(solid_pos, "solid_pos")
        check_tensor(solid_v, "solid_v")
        check_tensor(solid_q, "solid_q")
        check_tensor(solid_L_world, "solid_L_world")

        rho.append(rho_t.clone())
        u_list.append(u_t.clone())
        pressure_list.append(p_t.clone())
        solid_pos_list.append(solid_pos.clone())
        solid_q_list.append(solid_q.clone())
        solid_list.append(rotate_and_shift_mask(fluid_mask, solid_q, solid_pos).clone())

        p_visualize.extend(intermediates['p'])
        solid_visualize.extend([rotate_and_shift_mask(fluid_mask.detach().cpu(), q, pos) for q, pos in zip(intermediates['solid_q'], intermediates['solid_pos'])])
        u_visualize.extend(intermediates['u'])
        rho_visualize.extend(intermediates['rho'])
        solid_q_visualize.extend(intermediates['solid_q'])
        solid_pos_visualize.extend(intermediates['solid_pos'])

    state = {
        'rho_list': rho,
        'u_list': u_list,
        'fluid_mask': fluid_mask,
        'solid_list': solid_list,
        'pressure_list': pressure_list,
        'solid_pos_list': solid_pos_list,
        'solid_q_list': solid_q_list,
        'device': device,
    }

    if loss_specs is not None:
        loss, loss_dict = compute_total_loss(loss_specs, state)
    else:
        loss = torch.tensor(0.0, device=device)
        loss_dict = {}

    if do_visualize and step % 5 == 0:
        os.makedirs(save_dir, exist_ok=True)

        clean_plot = True

        visualization_resolution_scale = 2.0
        visualization_extent_scale = 1.4
        cutaway_side_offset_frac = 0.01
        slice_mask_viz, _ = build_visualization_mask_from_source(
            grid_shape=grid_shape,
            resolution_scale=visualization_resolution_scale,
            extent_scale=visualization_extent_scale,
            voxels_per_unit=voxels_per_unit,
            mask_sharpening=mask_sharpening,
            tile_size=tile_size,
            gaussian_means=gaussian_means,
            gaussian_scalings=gaussian_scalings,
            gaussian_rotations=gaussian_rotations,
            gaussian_opacities=gaussian_opacities,
        )
        slice_mask_viz_cpu = slice_mask_viz.detach().cpu()

        os.makedirs(save_dir / 'solid_mask_slices', exist_ok=True)
        visualize_solid_mask_slice(
            slice_mask_viz_cpu,
            save=True, slice_axis='x',
            filename=save_dir / 'solid_mask_slices' / f'slice_solid_mask_x_{step:04d}.png',
            clean_plot=clean_plot
        )
        visualize_solid_mask_slice(
            slice_mask_viz_cpu,
            save=True, slice_axis='y',
            filename=save_dir / 'solid_mask_slices' / f'slice_solid_mask_y_{step:04d}.png',
            clean_plot=clean_plot
        )
        visualize_solid_mask_slice(
            slice_mask_viz_cpu,
            save=True, slice_axis='z',
            filename=save_dir / 'solid_mask_slices' / f'slice_solid_mask_z_{step:04d}.png',
            clean_plot=clean_plot
        )
        post_processed_mask = 1.0 - slice_mask_viz
        post_processed_mask = 1.0 - prune_disconnected(post_processed_mask, low_tau=0.7, high_tau=0.9, return_soft=True)
        post_processed_mask = rotate_and_shift_mask(
            post_processed_mask.to(device),
            initial_q,
            original_pos.detach(),
        ).detach().cpu()

        os.makedirs(save_dir / 'solid_mask_slices_clean', exist_ok=True)
        visualize_solid_mask_slice(post_processed_mask, save=True, slice_axis='x', filename=save_dir / 'solid_mask_slices_clean' / f'slice_solid_mask_clean_{step:04d}.png', clean_plot=clean_plot)
        if pyvista_is_available():
            try:
                os.makedirs(save_dir / 'solid_mask_cutaways_mesh_pyvista', exist_ok=True)
                for cutaway_name, cutaway_offset in (
                    ('x_minus', -cutaway_side_offset_frac),
                    ('x', 0.0),
                    ('x_plus', cutaway_side_offset_frac),
                ):
                    visualize_solid_mask_cutaway_mesh_pyvista(
                        1.0 - slice_mask_viz_cpu,
                        slice_axis='x',
                        slice_offset_frac=cutaway_offset,
                        keep='negative',
                        save_path=save_dir / 'solid_mask_cutaways_mesh_pyvista' / f'cutaway_solid_mask_{cutaway_name}_{step:04d}.png',
                        view_angle=(20, 30),
                        cap_upsample=6,
                        show_context=False,
                        show_cut_plane=False,
                        removed_wire_opacity=0.07,
                        cut_plane_color=(1, 0, 0),
                        cut_plane_opacity=0.3,
                        cut_band=0.5,
                        show_removed_wire=False,
                    )
            except Exception as e:
                print(f"Error generating PyVista cutaways at step {step}: {e}")
        else:
            print("Skipping PyVista cutaway visualization: PyVista is not installed in the active environment.")


        os.makedirs(save_dir / 'vispy_sim_visualize', exist_ok=True)
        try:
            field_to_visualize = rho_visualize

            field_min = torch.min(torch.stack(field_to_visualize)).item()
            field_max = torch.max(torch.stack(field_to_visualize)).item()

            visualize_3d_fluid_animation_vispy(
                field_to_visualize,
                solid_visualize,
                threshold=0.0,
                fps=10,
                save_path=save_dir / 'vispy_sim_visualize' / f'sim_{step:04d}.mp4',
                rotate_camera=False,
                show_axis=False,
                save_frames=None,
                view_angle=(0, 90+180),
                field_min=field_min,
                field_max=field_max,
                show_bounds=False,
                show_velocity=False,
                velocity_list=u_visualize,
                velocity_spatial_order="dhw",
                velocity_component_order="xyz",
                velocity_stride=10,
                velocity_length_scale=.3,
                velocity_color=(0.1, 0.4, 1.0, 0.9),
                velocity_line_width=2,
                velocity_show_arrowheads=True,
                spatial_axis_map=(0, 1, 2),
                surface=surface,
            )
        except Exception as e:
            print(f"Error during fluid animation visualization: {e}")

        os.makedirs(save_dir / 'z_positions', exist_ok=True)
        os.makedirs(save_dir / 'x_positions', exist_ok=True)

        solid_z_positions = [sp[2].item() for sp in solid_pos_visualize]
        plt.plot(solid_z_positions)
        plt.xlabel('Time step')
        plt.ylabel('Solid Z Position')
        plt.title('Solid Z Position Over Time')
        plt.savefig(save_dir / 'z_positions' / f'solid_z_position_{step:04d}.png')
        plt.close()
        with open(save_dir / 'z_positions' / f'solid_z_positions_{step:04d}.json', 'w') as f:
            json.dump(solid_z_positions, f)

        solid_x_positions = [sp[0].item() for sp in solid_pos_visualize]
        plt.plot(solid_x_positions)
        plt.xlabel('Time step')
        plt.ylabel('Solid X Position')
        plt.title('Solid X Position Over Time')
        plt.savefig(save_dir / 'x_positions' / f'solid_x_position_{step:04d}.png')
        plt.close()
        with open(save_dir / 'x_positions' / f'solid_x_positions_{step:04d}.json', 'w') as f:
            json.dump(solid_x_positions, f)

        os.makedirs(save_dir / "stl_big", exist_ok=True)
        os.makedirs(save_dir / "stl_raw", exist_ok=True)

        try:
            verts, faces, normals, values = measure.marching_cubes(
                1 - fluid_mask.detach().cpu().numpy(), level=surface
            )
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            mesh.export(save_dir / "stl_raw" / f"output_{step:04d}.stl")
        except Exception as e:
            print(f"Error generating raw STL at step {step}: {e}")

        try:
            large_grid_shape = (D * 2, H * 2, W * 2)
            large_fluid_mask = gaussians_to_fluid_mask(
                gaussian_means,
                gaussian_scalings,
                gaussian_rotations,
                gaussian_opacities,
                grid_shape=large_grid_shape,
                voxels_per_unit=voxels_per_unit,
                tile_size=tile_size,
            )

            if large_fluid_mask is None:
                raise ValueError("Failed to build a large fluid mask for STL export.")
            large_fluid_mask = large_fluid_mask ** mask_sharpening
            verts, faces, normals, values = measure.marching_cubes(
                1 - large_fluid_mask.detach().cpu().numpy(), level=surface
            )
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            mesh.export(save_dir / "stl_big" / f"output_{step:04d}.stl")
        except Exception as e:
            print(f"Error generating big STL at step {step}: {e}")
            
    return loss
