try:
    from pxr import Usd, UsdGeom, UsdVol, Gf, Sdf
except ImportError:
    Usd = UsdGeom = UsdVol = Gf = Sdf = None

import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
import numpy as np
import os
from vispy import scene, app, visuals, use
from vispy.visuals.transforms import STTransform
# from skimage.measure import marching_cubes
from skimage.measure import marching_cubes
from skimage.transform import resize
import torch
from vispy import gloo
from vispy.gloo import gl
from scipy.ndimage import gaussian_filter
# import pyvista as pv

use('egl')


def _require_pyvista():
    headless = not os.environ.get("DISPLAY")
    if headless:
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for this visualization. Install it in the active environment."
        ) from exc
    if headless:
        try:
            pv.OFF_SCREEN = True
        except Exception:
            pass
        try:
            from pyvista import _vtk
            _vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass
    return pv


def pyvista_is_available():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return False
    return True


def create_interactive_3d_volume_with_solid(rho_list, solid_mask, output_html="interactive_volume_with_solid.html"):
    if go is None:
        raise ImportError(
            "Plotly is required for interactive HTML visualization and is not installed."
        )

    D, H, W = rho_list[0].shape
    X, Y, Z = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij')

    frames = []
    for t, rho in enumerate(rho_list):
        # Fluid volume
        fluid_volume = go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=rho.detach().cpu().numpy().flatten(),
            isomin=0.01,
            isomax=1.0,
            opacity=0.1,
            surface_count=10,
            colorscale="Viridis",
            name="fluid",
            showscale=False
        )

        # Solid isosurface (optional tweak: show near-zero for visual thickness)
        solid_surface = go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=solid_mask.detach().cpu().numpy().flatten(),
            isomin=0.01,
            isomax=0.05,
            opacity=0.3,
            surface_count=1,
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="solid"
        )

        frames.append(go.Frame(data=[fluid_volume, solid_surface], name=str(t)))

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0), transition=dict(duration=0))], label=str(k)) for k in range(len(rho_list))],
            active=0
        )]
    )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
    fig.write_html(output_html)
    print(f"Interactive 3D volume with solid saved to {output_html}")


import numpy as np
import imageio
from vispy import scene, app
from skimage.measure import marching_cubes
from tqdm import tqdm

def dump_gl_info():
    vendor   = gl.glGetString(gl.GL_VENDOR)
    renderer = gl.glGetString(gl.GL_RENDERER)
    version  = gl.glGetString(gl.GL_VERSION)
    shading  = gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)

    def _b(x):
        return x.decode("utf-8", "ignore") if x is not None else None

    print("GL_VENDOR  :", _b(vendor))
    print("GL_RENDERER:", _b(renderer))
    print("GL_VERSION :", _b(version))
    print("GLSL       :", _b(shading))

    max_3d = gl.glGetIntegerv(gl.GL_MAX_3D_TEXTURE_SIZE)
    max_tex = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
    print("GL_MAX_3D_TEXTURE_SIZE:", int(max_3d))
    print("GL_MAX_TEXTURE_SIZE   :", int(max_tex))


def visualize_3d_fluid_animation_vispy(rho_list, solid_mask_list, threshold=0.05, fps=10, save_path=None, view_angle=(30, 45), rotate_camera=False, show_axis=True, save_frames=None, field_min=0.0, field_max=1.0, particles=None, show_bounds=True, velocity_list=None, show_velocity=False, velocity_stride=4, velocity_scale=1.0, velocity_length_scale=1.0, velocity_color=(0.2, 0.2, 0.2, 0.6), velocity_line_width=2, velocity_show_arrowheads=False, velocity_min=0.0, velocity_max_arrows=4000, velocity_arrow_size=6, velocity_component_order="zyx", velocity_spatial_order="wdh", spatial_axis_map=(2, 1, 0), surface=0.5):
    """
    Visualize a 3D fluid animation with moving solids using Vispy.

    Args:
        rho_list: list of (D, H, W) torch tensors for fluid density at each timestep.
        solid_mask_list: list of (D, H, W) torch tensors for solid mask at each timestep.
        threshold: threshold for fluid volume visualization.
        fps: frames per second for animation/video.
        save_path: if not None, save animation to this path as a video.
        rotate_camera: if True, rotate camera during animation.
        show_axis: if True, show XYZ axes with labels.
        save_frames: if not None, save every `save_frames` frame as an image.
        velocity_list: optional list of (D, H, W, 3) velocity tensors.
        show_velocity: if True, draw velocity arrows when velocity_list is provided.
        velocity_stride: stride for subsampling velocity vectors on the grid.
        velocity_scale: scale factor applied to velocity vectors for visualization.
        velocity_length_scale: extra multiplier for visualizing longer lines at coarser strides.
        velocity_color: RGBA color for velocity arrows/lines.
        velocity_line_width: line width in pixels for velocity segments.
        velocity_show_arrowheads: if True, add arrowheads at the tips.
        velocity_min: minimum velocity magnitude to display.
        velocity_max_arrows: cap on number of arrows to draw after filtering.
        velocity_arrow_size: arrowhead size (when supported by Vispy).
        velocity_component_order: string of 'xyz' order for velocity components in velocity_list.
        velocity_spatial_order: spatial axis order for velocity_list (e.g., 'dhw', 'hwd', 'wdh').
        spatial_axis_map: tuple mapping data axes (D,H,W) to Vispy (x,y,z).
    """
    T = len(rho_list)
    D, H, W = rho_list[0].shape

    scale = 5  # pixels per voxel (adjust as needed)
    if len(spatial_axis_map) != 3 or set(spatial_axis_map) != {0, 1, 2}:
        raise ValueError("spatial_axis_map must be a permutation of (0, 1, 2).")
    vispy_dims = np.array([D, H, W], dtype=float)[list(spatial_axis_map)]

    def round_up_16(x):
        return ((x + 15) // 16) * 16

    width = round_up_16(min(int(max(D, H) * scale), 1200))
    height = round_up_16(min(int(W * scale), 1200))

    canvas = scene.SceneCanvas(keys='interactive', show=(save_path is None), bgcolor='white', size=(width, height))
    view = canvas.central_widget.add_view()

        # Optional visible bounding box
    if show_bounds:
        box = scene.visuals.Box(width=vispy_dims[0], height=vispy_dims[1], depth=vispy_dims[2], color=(0, 0, 0, 0.3), edge_color=(0, 0, 0, 0.5), parent=view.scene)
        box.transform = STTransform(translate=(vispy_dims[0] / 2, vispy_dims[1] / 2, vispy_dims[2] / 2))
        box.set_gl_state(depth_test=True, blend=True)


    view.camera = scene.cameras.TurntableCamera(
        azimuth=45+45,
        elevation=0,
        distance=550,
        center=(vispy_dims[0] / 2, vispy_dims[1] / 2, vispy_dims[2] / 2)
    )

    diag = (D**2 + H**2 + W**2) ** 0.5
    view.camera.distance = 1.0 * diag

    # make it easier to see the fluid, push it up towards 1.0 if it is close
    # rho_list = [rho**0.2 for rho in rho_list]

    # Initial fluid volume
    volume_axes = (spatial_axis_map[2], spatial_axis_map[1], spatial_axis_map[0])
    rho0 = np.transpose(rho_list[0].cpu().numpy(), axes=volume_axes)
    # apply gaussian smoothing for better visualization

    # rho0 = gaussian_filter(rho0, sigma=0.5)

    vol = scene.visuals.Volume(rho0, parent=view.scene, threshold=threshold, clim=(field_min, field_max))
    # vol.cmap = 'PuBuGn' # or blues
    vol.cmap = 'blues'

    # Initial solid mesh (if any)
    solid_mesh = None
    def update_solid_mesh(solid_mask_t):
        nonlocal solid_mesh


        solid_np = solid_mask_t.cpu().numpy()
        solid_np = gaussian_filter(solid_np, sigma=0.5)
        if (solid_np < 0.5).any():
            verts, faces, _, _ = marching_cubes(1-solid_np, level=surface)
            verts = verts[:, list(spatial_axis_map)]
            if solid_mesh is None:
                solid_mesh = scene.visuals.Mesh(vertices=verts, faces=faces, color=(.4, .4, .4, 0.7), shading='smooth', parent=view.scene)
                solid_mesh.set_gl_state(depth_test=False, blend=True, cull_face=True)
            else:
                solid_mesh.set_data(vertices=verts, faces=faces)
                solid_mesh.visible = True
        else:
            if solid_mesh is not None:
                solid_mesh.visible = False

    # Show first solid
    update_solid_mesh(solid_mask_list[0])

    if show_axis:
        # Add labeled XYZ axes with better visibility
        # Add XYZ axis and labels programmatically
        axis_length = 20
        axis_center = np.array([0,0,0])

        axis = scene.visuals.XYZAxis(parent=view.scene)
        axis.transform = STTransform(translate=axis_center, scale=(axis_length, axis_length, axis_length))
        axis.set_gl_state(depth_test=False, blend=True)  # Always on top

        # Axis directions (unit vectors)
        directions = np.eye(3)
        labels = ['X', 'Y', 'Z']
        label_objs = []
        tick_objs = []
        tick_label_objs = []
        n_ticks = 5
        for i, label in enumerate(labels):
            pos = axis_center + directions[i] * axis_length
            label_obj = scene.visuals.Text(
                label,
                pos=pos,
                color='white',
                font_size=2000,
                parent=view.scene,
                anchor_x='center',
                anchor_y='center'
            )
            label_obj.set_gl_state(depth_test=False, blend=True)
            label_objs.append(label_obj)

            # Add tick marks and numbers
            tick_positions = np.linspace(0, axis_length, n_ticks+1)[1:]
            for tick in tick_positions:
                tick_start = axis_center + directions[i] * tick
                # Tick mark: short line perpendicular to axis
                perp = directions[(i+1)%3] * 1.0  # 1.0 is tick length
                tick_line = scene.visuals.Line(
                    pos=np.array([tick_start - perp*0.5, tick_start + perp*0.5]),
                    color='white',
                    width=2,
                    parent=view.scene,
                )
                tick_objs.append(tick_line)
                # Tick label
                tick_label = scene.visuals.Text(
                    f"{int(tick)}",
                    pos=tick_start + perp*1.5,
                    color='white',
                    font_size=1000,
                    parent=view.scene,
                    anchor_x='center',
                    anchor_y='center'
                )
                tick_label.set_gl_state(depth_test=False, blend=True)
                tick_label_objs.append(tick_label)

    if show_velocity:
        if velocity_list is None:
            raise ValueError("show_velocity=True requires velocity_list to be provided.")
        if len(velocity_list) != T:
            raise ValueError("velocity_list must match rho_list length.")

    velocity_visual = None
    velocity_arrow_visual = None

    def _to_numpy(field):
        if field is None:
            return None
        if torch.is_tensor(field):
            return field.detach().cpu().numpy()
        return np.asarray(field)

    def _prepare_velocity(u_t):
        u_np = _to_numpy(u_t)
        if u_np is None:
            return None, None
        if u_np.ndim != 4:
            raise ValueError("velocity_list entries must be 4D arrays shaped (D, H, W, 3) or (3, D, H, W).")
        if u_np.shape[-1] == 3:
            u_dhw3 = u_np
        elif u_np.shape[0] == 3:
            u_dhw3 = np.moveaxis(u_np, 0, -1)
        else:
            raise ValueError("velocity_list entries must have a 3-component velocity axis.")

        order = velocity_spatial_order.lower()
        if set(order) != {"d", "h", "w"} or len(order) != 3:
            raise ValueError("velocity_spatial_order must be a permutation of 'dhw'.")
        axis_map = {"d": None, "h": None, "w": None}
        spatial_shape = {"d": D, "h": H, "w": W}
        for i, axis in enumerate(order):
            axis_map[axis] = i
        target_shape = (spatial_shape["d"], spatial_shape["h"], spatial_shape["w"])
        permute = (axis_map["d"], axis_map["h"], axis_map["w"])
        if tuple(u_dhw3.shape[:3][i] for i in permute) == target_shape:
            u_dhw3 = np.transpose(u_dhw3, axes=permute + (3,))
        else:
            # Try all permutations to auto-recover if order is ambiguous.
            matched = []
            for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                if tuple(u_dhw3.shape[:3][i] for i in perm) == target_shape:
                    matched.append(perm)
            if len(matched) == 1:
                u_dhw3 = np.transpose(u_dhw3, axes=matched[0] + (3,))
            else:
                raise ValueError(
                    "velocity_list spatial shape does not match rho_list. "
                    "Set velocity_spatial_order to match your velocity array layout."
                )

        stride = max(int(velocity_stride), 1)
        z_idx = np.arange(0, D, stride)
        y_idx = np.arange(0, H, stride)
        x_idx = np.arange(0, W, stride)
        zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing='ij')
        idx_map = {0: zz, 1: yy, 2: xx}
        pos = np.stack([idx_map[axis] for axis in spatial_axis_map], axis=-1).reshape(-1, 3)

        u_sample = u_dhw3[np.ix_(z_idx, y_idx, x_idx)]
        order = velocity_component_order.lower()
        if set(order) != {"x", "y", "z"} or len(order) != 3:
            raise ValueError("velocity_component_order must be a permutation of 'xyz'.")
        comp_idx = {axis: i for i, axis in enumerate(order)}
        vec = u_sample[..., [comp_idx["x"], comp_idx["y"], comp_idx["z"]]].reshape(-1, 3)

        mag = np.linalg.norm(vec, axis=1)
        if velocity_min > 0.0:
            keep = mag >= velocity_min
            pos = pos[keep]
            vec = vec[keep]

        if velocity_max_arrows and pos.shape[0] > velocity_max_arrows:
            idx = np.linspace(0, pos.shape[0] - 1, velocity_max_arrows).astype(int)
            pos = pos[idx]
            vec = vec[idx]

        vec = vec * (velocity_scale * velocity_length_scale)
        return pos, vec

    def _update_velocity_visual(u_t):
        nonlocal velocity_visual, velocity_arrow_visual
        if not show_velocity:
            return
        pos, vec = _prepare_velocity(u_t)
        if pos is None or pos.size == 0:
            if velocity_visual is not None:
                velocity_visual.visible = False
            return

        start = pos
        end = pos + vec
        line_pos = np.empty((start.shape[0] * 2, 3), dtype=start.dtype)
        line_pos[0::2] = start
        line_pos[1::2] = end
        if velocity_visual is None:
            velocity_visual = scene.visuals.Line(
                pos=line_pos,
                color=velocity_color,
                connect='segments',
                width=velocity_line_width,
                parent=view.scene
            )
            velocity_visual.set_gl_state(depth_test=False, blend=True)
        else:
            velocity_visual.set_data(pos=line_pos, color=velocity_color)
            velocity_visual.visible = True

        if velocity_show_arrowheads:
            arrows = np.concatenate([start, end], axis=1)
            if velocity_arrow_visual is None:
                velocity_arrow_visual = scene.visuals.Arrow(
                    pos=line_pos,
                    color=velocity_color,
                    connect='segments',
                    arrows=arrows,
                    arrow_size=velocity_arrow_size,
                    arrow_color=velocity_color,
                    parent=view.scene
                )
                velocity_arrow_visual.set_gl_state(depth_test=False, blend=True)
            else:
                velocity_arrow_visual.set_data(pos=line_pos, color=velocity_color, arrows=arrows)
                velocity_arrow_visual.visible = True
        elif velocity_arrow_visual is not None:
            velocity_arrow_visual.visible = False

    if show_velocity:
        _update_velocity_visual(velocity_list[0])

    if save_path is not None:
        writer = imageio.get_writer(save_path, fps=fps)
        for t in range(T):
            rho_t = np.transpose(rho_list[t].cpu().numpy(), axes=volume_axes)
            vol.set_data(rho_t)
            update_solid_mesh(solid_mask_list[t])
            if show_velocity:
                _update_velocity_visual(velocity_list[t])

            # Update camera position for smooth movement
            if rotate_camera:
                angle = -360 / 4 * (t / T)
            else:
                angle = view_angle[1]  # Use provided azimuth from view_angle
            view.camera.azimuth = angle
            view.camera.elevation = view_angle[0]  # Use provided elevation from view_angle

            canvas.render()
            frame = canvas.render()
            frame = np.asarray(frame)
            writer.append_data(frame)
            if save_frames is not None and t % save_frames == 0:
                parent_dir = os.path.dirname(save_path)
                imageio.imwrite(os.path.join(parent_dir, f"frame_{t:04d}.png"), frame)
        writer.close()
    else:
        frame_idx = {'t': 0}
        def update(ev):
            t = frame_idx['t']
            rho_t = np.transpose(rho_list[t % T].cpu().numpy(), axes=volume_axes)
            vol.set_data(rho_t)
            update_solid_mesh(solid_mask_list[t % T])
            if show_velocity:
                _update_velocity_visual(velocity_list[t % T])


            canvas.update()
            frame_idx['t'] += 1
        timer = app.Timer(interval=1.0 / fps, connect=update, start=True)
        app.run()


def visualize_3d_fluid_vispy(
    rho_list,
    solid_mask_list,
    *,
    threshold=0.05,
    fps=10,
    save_path=None,
    field_clim=None,          # (min, max) or None
    view_angle=(30, 45),      # (elevation, azimuth)
    spatial_axis_map=(2, 1, 0),
    save_every=None,          # save every N frames as png
    frames_dir=None,          # directory for pngs; default рядом с save_path
    canvas_scale=5,           # pixels per voxel-ish
    max_canvas=1200,
    smooth_solid_sigma=0.5,
):
    """
    Minimal 3D Vispy visualization: volume (rho) + solid mesh, with optional video and/or frame saving.

    rho_list: list of (D,H,W) torch tensors/arrays
    solid_mask_list: list of (D,H,W) torch tensors/arrays (1=solid, 0=empty or vice-versa; see below)
    """
    T = len(rho_list)
    if T == 0:
        raise ValueError("rho_list is empty.")
    if len(solid_mask_list) != T:
        raise ValueError("solid_mask_list must have same length as rho_list.")

    # shapes
    rho0_t = rho_list[0]
    if torch.is_tensor(rho0_t):
        D, H, W = rho0_t.shape
    else:
        D, H, W = np.asarray(rho0_t).shape

    if len(spatial_axis_map) != 3 or set(spatial_axis_map) != {0, 1, 2}:
        raise ValueError("spatial_axis_map must be a permutation of (0,1,2).")

    vispy_dims = np.array([D, H, W], dtype=float)[list(spatial_axis_map)]

    def round_up_16(x: int) -> int:
        return ((x + 15) // 16) * 16

    width = round_up_16(min(int(max(D, H) * canvas_scale), max_canvas))
    height = round_up_16(min(int(W * canvas_scale), max_canvas))

    canvas = scene.SceneCanvas(
        keys="interactive",
        show=(save_path is None),
        bgcolor="white",
        size=(width, height),
    )

    # dump_gl_info()
    view = canvas.central_widget.add_view()

    # Camera: fixed, turntable
    view.camera = scene.cameras.TurntableCamera(
        azimuth=float(view_angle[1]),
        elevation=float(view_angle[0]),
        center=(vispy_dims[0] / 2, vispy_dims[1] / 2, vispy_dims[2] / 2),
    )
    diag = float((D * D + H * H + W * W) ** 0.5)
    view.camera.distance = 1.0 * diag

    # volume axes: vispy expects (Z,Y,X) style; you were using:
    volume_axes = (spatial_axis_map[2], spatial_axis_map[1], spatial_axis_map[0])

    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # Initial volume
    rho0 = np.transpose(to_numpy(rho_list[0]), axes=volume_axes)
    if field_clim is None:
        # safe default: auto from first frame
        vmin = float(np.min(rho0))
        vmax = float(np.max(rho0))
        field_clim = (vmin, vmax)

    print("Original (D,H,W):", D, H, W)
    print("Volume array shape:", rho0.shape)


    vol = scene.visuals.Volume(rho0, parent=view.scene, threshold=threshold, clim=field_clim)
    vol.cmap = "blues"

    # Solid mesh (marching cubes)
    solid_mesh = None

    def update_solid_mesh(solid_mask_t):
        nonlocal solid_mesh
        solid_np = to_numpy(solid_mask_t).astype(np.float32)

        # optional smoothing for nicer marching cubes
        if smooth_solid_sigma and smooth_solid_sigma > 0:
            solid_np = gaussian_filter(solid_np, sigma=float(smooth_solid_sigma))

        # Your original did marching_cubes(1-solid_np, level=0.5)
        # which assumes solid_np ~1 means "solid". Keep same behavior:
        iso_field = 1.0 - solid_np

        # if everything is solid (solid_np==1 everywhere) => iso_field==0 everywhere => no surface
        if not (iso_field > 0.5).any() and not (iso_field < 0.5).any():
            if solid_mesh is not None:
                solid_mesh.visible = False
            return

        # Only attempt if there's a crossing
        if (iso_field < 0.5).any() and (iso_field > 0.5).any():
            verts, faces, _, _ = marching_cubes(iso_field, level=0.5)
            verts = verts[:, list(spatial_axis_map)]

            if solid_mesh is None:
                solid_mesh = scene.visuals.Mesh(
                    vertices=verts,
                    faces=faces,
                    color=(0.4, 0.4, 0.4, 0.7),
                    shading="smooth",
                    parent=view.scene,
                )
                solid_mesh.set_gl_state(depth_test=False, blend=True, cull_face=True)
            else:
                solid_mesh.set_data(vertices=verts, faces=faces)
                solid_mesh.visible = True
        else:
            if solid_mesh is not None:
                solid_mesh.visible = False

    update_solid_mesh(solid_mask_list[0])

    # Output setup
    writer = None
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        writer = imageio.get_writer(str(save_path), fps=fps)

    if save_every is not None and save_every > 0:
        if frames_dir is None:
            if save_path is None:
                frames_dir = "frames"
            else:
                frames_dir = os.path.join(os.path.dirname(str(save_path)), "frames")
        os.makedirs(frames_dir, exist_ok=True)

    def render_frame():
        frame = canvas.render()
        return np.asarray(frame)

    # Render loop: offline if saving video, interactive otherwise
    if writer is not None:
        for t in range(T):
            rho_t = np.transpose(to_numpy(rho_list[t]), axes=volume_axes)
            vol.set_data(rho_t)
            update_solid_mesh(solid_mask_list[t])

            frame = render_frame()
            writer.append_data(frame)

            if save_every is not None and (t % save_every == 0):
                imageio.imwrite(os.path.join(frames_dir, f"frame_{t:04d}.png"), frame)

        writer.close()
    else:
        frame_idx = {"t": 0}

        def update(_ev):
            t = frame_idx["t"] % T
            rho_t = np.transpose(to_numpy(rho_list[t]), axes=volume_axes)
            vol.set_data(rho_t)
            update_solid_mesh(solid_mask_list[t])

            canvas.update()
            frame_idx["t"] += 1

        timer = app.Timer(interval=1.0 / fps, connect=update, start=True)
        app.run()




def visualize_3d_fluid_pyvista(
    rho_list,
    solid_mask_list,
    *,
    threshold=0.05,
    fps=10,
    save_path=None,
    field_clim=None,          # (min, max) or None
    view_angle=(30, 45),      # (elevation, azimuth) in degrees
    spatial_axis_map=(2, 1, 0),
    save_every=None,          # save every N frames as png
    frames_dir=None,          # directory for pngs
    shade=True,
):
    """
    Minimal PyVista renderer: volume rho + solid surface, optional movie + frames.

    rho_list: list of (D,H,W) torch tensors/arrays
    solid_mask_list: list of (D,H,W) torch tensors/arrays, where ~1 is solid (same convention as your vispy code)
    spatial_axis_map: maps (D,H,W) -> (x,y,z) axes order, e.g. (0,1,2) means x=D, y=H, z=W
    """
    pv = _require_pyvista()
    T = len(rho_list)
    if T == 0:
        raise ValueError("rho_list is empty.")
    if len(solid_mask_list) != T:
        raise ValueError("solid_mask_list must match rho_list length.")

    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    rho0_dhw = to_numpy(rho_list[0])
    if rho0_dhw.ndim != 3:
        raise ValueError("rho_list entries must be 3D arrays shaped (D,H,W).")
    D, H, W = rho0_dhw.shape

    if len(spatial_axis_map) != 3 or set(spatial_axis_map) != {0, 1, 2}:
        raise ValueError("spatial_axis_map must be a permutation of (0,1,2).")

    # Reorder to xyz for VTK
    def dhw_to_xyz(a_dhw):
        a_xyz = np.transpose(a_dhw, axes=spatial_axis_map)
        return np.ascontiguousarray(a_xyz, dtype=np.float32)

    rho0 = dhw_to_xyz(rho0_dhw)
    nx, ny, nz = rho0.shape  # xyz

    if field_clim is None:
        field_clim = (float(np.min(rho0)), float(np.max(rho0)))

    # Build a UniformGrid with CELL data (voxel data). For VTK cell data, dims are +1.
    grid = pv.UniformGrid()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)   # points dims
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)

    # VTK cell ordering is Fortran-like; ravel(order="F") is the usual trick
    grid.cell_data["rho"] = rho0.ravel(order="F")

    # Solid grid (store iso_field = 1 - solid_mask, contour at 0.5)
    solid_grid = pv.UniformGrid()
    solid_grid.dimensions = (nx + 1, ny + 1, nz + 1)
    solid_grid.origin = (0.0, 0.0, 0.0)
    solid_grid.spacing = (1.0, 1.0, 1.0)

    # Create a simple opacity transfer function:
    #  - zero below threshold
    #  - ramps up above threshold
    vmin, vmax = field_clim
    n_tf = 256
    opacity = np.linspace(0.0, 1.0, n_tf, dtype=np.float32)
    if vmax > vmin:
        frac = (threshold - vmin) / (vmax - vmin)
        cut = int(np.clip(frac, 0.0, 1.0) * (n_tf - 1))
        opacity[:cut] = 0.0
        # optionally soften the ramp a bit
        opacity[cut:] = np.power(opacity[cut:], 1.5)

    offscreen = save_path is not None
    pl = pv.Plotter(off_screen=offscreen, window_size=(1200, 900))
    pl.set_background("white")

    # Add volume once
    vol_actor = pl.add_volume(
        grid,
        scalars="rho",
        clim=field_clim,
        opacity=opacity,
        shade=shade,
    )

    # Add first solid surface
    solid_actor = None

    def update_solid_actor(solid_mask_dhw):
        nonlocal solid_actor
        solid_xyz = dhw_to_xyz(to_numpy(solid_mask_dhw))
        iso_field = 1.0 - solid_xyz  # keep your original convention
        solid_grid.cell_data["iso"] = iso_field.ravel(order="F")

        # Extract iso-surface at 0.5
        surf = solid_grid.contour(isosurfaces=[0.5], scalars="iso")

        # Replace actor each frame (topology changes)
        if solid_actor is not None:
            pl.remove_actor(solid_actor)
            solid_actor = None

        if surf.n_points > 0 and surf.n_cells > 0:
            solid_actor = pl.add_mesh(
                surf,
                color=(0.4, 0.4, 0.4),
                opacity=0.7,
                smooth_shading=True,
            )

    update_solid_actor(solid_mask_list[0])

    # Fixed camera from (elevation, azimuth)
    center = np.array([nx / 2.0, ny / 2.0, nz / 2.0], dtype=np.float32)
    diag = float(np.linalg.norm([nx, ny, nz]))
    dist = 1.2 * diag

    elev_deg, azim_deg = float(view_angle[0]), float(view_angle[1])
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    cam_pos = center + dist * np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
        dtype=np.float32,
    )
    pl.camera_position = [cam_pos.tolist(), center.tolist(), [0, 0, 1]]

    # Output setup
    if save_every is not None and save_every > 0:
        if frames_dir is None:
            frames_dir = os.path.join(os.path.dirname(str(save_path)) if save_path else ".", "frames")
        os.makedirs(frames_dir, exist_ok=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        pl.open_movie(str(save_path), framerate=fps)

    # Render loop
    for t in range(T):
        rho_t = dhw_to_xyz(to_numpy(rho_list[t]))
        grid.cell_data["rho"] = rho_t.ravel(order="F")

        # This updates the scalars on the dataset used by the volume mapper
        # (works reliably because the same `grid` instance is kept)
        pl.update_scalars(grid.cell_data["rho"], mesh=grid, render=False)

        update_solid_actor(solid_mask_list[t])

        pl.render()

        if save_path is not None:
            pl.write_frame()

        if save_every is not None and save_every > 0 and (t % save_every == 0):
            pl.screenshot(os.path.join(frames_dir, f"frame_{t:04d}.png"))

    if save_path is not None:
        pl.close()
    else:
        pl.show()


def visualize_solid_mask_cutaway_pyvista(
    solid_mask,
    *,
    slice_axis='x',
    slice_idx=None,
    keep='positive',
    save_path=None,
    view_angle=(20, 30),
    smooth_sigma=0.6,
    volume_cmap='bone',
    volume_opacity=None,
    highlight_cut=True,
    cut_highlight_width=1,
    cut_highlight_cmap='gist_heat',
    cut_highlight_opacity=None,
    shade=True,
    show_context=False,
    context_opacity=0.08,
    surface_opacity=0.18,
    background='white',
    opacity_unit_distance=None,
    window_size=(1000, 1000),
):
    """
    Render a half-object voxel cutaway with PyVista volume rendering.

    This keeps the same axis convention as the Vispy cutaway helper:
    x -> axis 0, y -> axis 1, z -> axis 2 for a (D, H, W) array.
    """
    pv = _require_pyvista()

    def _to_numpy(vol):
        if torch is not None and isinstance(vol, torch.Tensor):
            return vol.detach().cpu().numpy()
        return np.asarray(vol)

    def _make_image_data(vol_xyz, cut_highlight_xyz=None):
        nx, ny, nz = vol_xyz.shape
        grid = pv.ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.origin = (0.0, 0.0, 0.0)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.cell_data["occupancy"] = np.ascontiguousarray(vol_xyz, dtype=np.float32).ravel(order="F")
        if cut_highlight_xyz is not None:
            grid.cell_data["cut_highlight"] = np.ascontiguousarray(
                cut_highlight_xyz,
                dtype=np.float32,
            ).ravel(order="F")
        return grid

    def _contour_cell_volume(dataset, level=0.5):
        try:
            point_dataset = dataset.cell_data_to_point_data(pass_cell_data=True)
        except TypeError:
            point_dataset = dataset.cell_data_to_point_data()
        return point_dataset.contour(isosurfaces=[level], scalars="occupancy")

    solid_np = _to_numpy(solid_mask).astype(np.float32)
    if solid_np.ndim != 3:
        raise ValueError("solid_mask must be a 3D array shaped (D, H, W).")

    slice_axis = str(slice_axis).lower()
    axis_lookup = {'x': 0, 'y': 1, 'z': 2}
    if slice_axis not in axis_lookup:
        raise ValueError("slice_axis must be one of 'x', 'y', or 'z'.")
    axis_idx = axis_lookup[slice_axis]
    axis_len = solid_np.shape[axis_idx]
    if axis_len < 2:
        raise ValueError("Cutaway visualization requires at least 2 voxels along the slice axis.")

    if slice_idx is None:
        slice_idx = axis_len // 2
    slice_idx = int(np.clip(slice_idx, 1, axis_len - 1))

    keep = str(keep).lower()
    if keep not in {'positive', 'negative'}:
        raise ValueError("keep must be 'positive' or 'negative'.")

    if smooth_sigma and smooth_sigma > 0:
        solid_vis = gaussian_filter(solid_np, sigma=float(smooth_sigma))
    else:
        solid_vis = solid_np

    # Keep axis semantics aligned with the Vispy cutaway: x/y/z map to array axes 0/1/2.
    volume_xyz = np.ascontiguousarray(solid_vis, dtype=np.float32)
    cut_highlight_xyz = None
    if highlight_cut:
        band = max(int(np.ceil(float(cut_highlight_width))), 1)
        cut_highlight_xyz = np.zeros_like(volume_xyz, dtype=np.float32)
        slab_slices = [slice(None)] * 3
        if keep == 'positive':
            slab_slices[axis_idx] = slice(slice_idx, min(slice_idx + band, axis_len))
        else:
            slab_slices[axis_idx] = slice(max(slice_idx - band, 0), slice_idx)
        slab_slices = tuple(slab_slices)
        cut_highlight_xyz[slab_slices] = volume_xyz[slab_slices]

    grid = _make_image_data(volume_xyz, cut_highlight_xyz=cut_highlight_xyz)

    nx, ny, nz = volume_xyz.shape
    extent = [0, nx, 0, ny, 0, nz]
    lo_idx, hi_idx = ((0, 1), (2, 3), (4, 5))[axis_idx]
    if keep == 'positive':
        extent[lo_idx] = slice_idx
    else:
        extent[hi_idx] = slice_idx
    cutaway = grid.extract_subset(extent)

    if volume_opacity is None:
        volume_opacity = [0.0, 0.0, 0.04, 0.16, 0.42, 0.8, 1.0]
    if opacity_unit_distance is None:
        opacity_unit_distance = float(np.mean(cutaway.spacing))

    off_screen = save_path is not None
    pl = pv.Plotter(off_screen=off_screen, window_size=window_size)
    pl.set_background(background)

    if show_context:
        context_surf = _contour_cell_volume(grid, level=0.5)
        if context_surf.n_points > 0 and context_surf.n_cells > 0:
            pl.add_mesh(
                context_surf,
                color=(0.72, 0.74, 0.78),
                opacity=context_opacity,
                smooth_shading=True,
            )

    volume_actor = pl.add_volume(
        cutaway,
        scalars="occupancy",
        cmap=volume_cmap,
        clim=(0.0, 1.0),
        opacity=volume_opacity,
        shade=shade,
        show_scalar_bar=False,
        opacity_unit_distance=opacity_unit_distance,
    )
    if hasattr(volume_actor, "prop"):
        volume_actor.prop.interpolation_type = 'linear'

    if highlight_cut and "cut_highlight" in cutaway.cell_data:
        if cut_highlight_opacity is None:
            cut_highlight_opacity = [0.0, 0.0, 0.0, 1., 1., 1.]
        cut_highlight_actor = pl.add_volume(
            cutaway,
            scalars="cut_highlight",
            cmap=cut_highlight_cmap,
            clim=(0.0, 1.0),
            opacity=cut_highlight_opacity,
            shade=shade,
            show_scalar_bar=False,
            opacity_unit_distance=opacity_unit_distance,
        )
        if hasattr(cut_highlight_actor, "prop"):
            cut_highlight_actor.prop.interpolation_type = 'linear'

    shell = _contour_cell_volume(cutaway, level=0.5)
    if shell.n_points > 0 and shell.n_cells > 0:
        pl.add_mesh(
            shell,
            color=(0.18, 0.20, 0.24),
            opacity=surface_opacity,
            smooth_shading=True,
        )

    bounds = cutaway.bounds
    center = np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=np.float32,
    )
    size = np.array(
        [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ],
        dtype=np.float32,
    )
    diag = float(np.linalg.norm(size))
    if diag <= 0:
        diag = float(np.sqrt(nx ** 2 + ny ** 2 + nz ** 2))

    elev_deg, azim_deg = float(view_angle[0]), float(view_angle[1])
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    cam_pos = center + 1.55 * diag * np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
        dtype=np.float32,
    )
    pl.camera_position = [cam_pos.tolist(), center.tolist(), [0, 0, 1]]
    pl.camera.zoom(1.1)

    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
        pl.show(screenshot=str(save_path), auto_close=True)
    else:
        pl.show()


def visualize_solid_mask_cutaway_mesh_pyvista(
    solid_mask,
    *,
    slice_axis='x',
    slice_idx=None,
    slice_offset_frac=0.0,
    keep='positive',
    iso_level=0.5,
    save_path=None,
    view_angle=(20, 30),
    smooth_sigma=0.6,
    cut_band=0.75,
    cap_upsample=4,
    show_cut_plane=True,
    cut_plane_opacity=0.22,
    cut_plane_color=(0.96, 0.72, 0.52),
    cut_plane_scale=1.35,
    show_cut_outline=True,
    cut_outline_color=(0.12, 0.12, 0.12),
    cut_outline_width=3.0,
    cut_outline_opacity=1.0,
    show_removed_wire=True,
    removed_wire_opacity=0.14,
    removed_wire_color=(0.45, 0.48, 0.54),
    removed_wire_width=1.0,
    show_context=False,
    context_opacity=0.08,
    outer_color=(0.62, 0.64, 0.68),
    cut_color=(0.88, 0.42, 0.22),
    cut_pattern='hatch',
    cut_checker_size=1.5,
    cut_checker_dark=(0.67, 0.33, 0.0),
    cut_checker_light=(1.0, 0.5, 0.0),
    cut_hatch_spacing=1.0,
    cut_hatch_angle_deg=35.0,
    cut_hatch_line_fraction=0.28,
    cut_hatch_dark=(0.30, 0.30, 0.30),
    cut_hatch_light=(0.80, 0.80, 0.80),
    background='white',
    camera_distance_scale=0.4,
    window_size=(1000, 1000),
):
    """
    Render a marching-cubes mesh cutaway with PyVista.

    This builds the full marching-cubes surface first, then clips the mesh
    with a plane. Clipping the mesh instead of zeroing half the voxel field
    produces a cleaner cut boundary.
    """
    pv = _require_pyvista()

    def _to_numpy(vol):
        if torch is not None and isinstance(vol, torch.Tensor):
            return vol.detach().cpu().numpy()
        return np.asarray(vol)

    def _extract_mesh(vol):
        vmin = float(vol.min())
        vmax = float(vol.max())
        if not (vmin < iso_level < vmax):
            return None, None
        verts, faces, _, _ = marching_cubes(vol, level=iso_level)
        return verts.astype(np.float32), faces.astype(np.int64)

    def _to_polydata(verts, faces, point_rgb=None):
        face_array = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
        ).ravel()
        poly = pv.PolyData(verts, face_array)
        if point_rgb is not None:
            poly["colors"] = np.clip(point_rgb * 255.0, 0, 255).astype(np.uint8)
        return poly

    def _normalize_surface(mesh):
        if mesh is None:
            return None
        if hasattr(mesh, "extract_surface"):
            try:
                mesh = mesh.extract_surface(algorithm="dataset_surface")
            except TypeError:
                mesh = mesh.extract_surface()
        if hasattr(mesh, "triangulate"):
            mesh = mesh.triangulate()
        if hasattr(mesh, "clean"):
            mesh = mesh.clean()
        if getattr(mesh, "n_points", 0) == 0 or getattr(mesh, "n_cells", 0) == 0:
            return None
        return mesh

    def _clip_half(poly, plane_coord_value, keep_side):
        poly = _normalize_surface(poly)
        if poly is None:
            return None
        keep_side = str(keep_side).lower()
        if keep_side not in {'negative', 'positive'}:
            raise ValueError("keep_side must be 'negative' or 'positive'.")

        mesh_points = np.asarray(poly.points, dtype=np.float32)
        mesh_faces = np.asarray(poly.faces)
        if mesh_points.size == 0 or mesh_faces.size == 0:
            return None
        mesh_faces = mesh_faces.reshape(-1, 4)[:, 1:]

        side_tolerance = 1e-6

        def _is_inside(signed_distance):
            if keep_side == 'negative':
                return signed_distance <= side_tolerance
            return signed_distance >= -side_tolerance

        def _intersect_edge(p0, p1, d0, d1):
            denom = d1 - d0
            if abs(float(denom)) < 1e-8:
                point = 0.5 * (p0 + p1)
            else:
                t = float(np.clip(-d0 / denom, 0.0, 1.0))
                point = p0 + t * (p1 - p0)
            point = np.asarray(point, dtype=np.float32)
            point[axis_idx] = float(plane_coord_value)
            return point

        clipped_points: list[np.ndarray] = []
        clipped_faces: list[list[int]] = []

        for face in mesh_faces:
            polygon_points = [mesh_points[idx] for idx in face]
            polygon_distances = [
                float(point[axis_idx] - plane_coord_value)
                for point in polygon_points
            ]

            clipped_polygon: list[np.ndarray] = []
            prev_point = polygon_points[-1]
            prev_distance = polygon_distances[-1]
            prev_inside = _is_inside(prev_distance)

            for curr_point, curr_distance in zip(polygon_points, polygon_distances):
                curr_inside = _is_inside(curr_distance)
                if curr_inside:
                    if not prev_inside:
                        clipped_polygon.append(
                            _intersect_edge(prev_point, curr_point, prev_distance, curr_distance)
                        )
                    clipped_polygon.append(np.asarray(curr_point, dtype=np.float32))
                elif prev_inside:
                    clipped_polygon.append(
                        _intersect_edge(prev_point, curr_point, prev_distance, curr_distance)
                    )

                prev_point = curr_point
                prev_distance = curr_distance
                prev_inside = curr_inside

            if len(clipped_polygon) < 3:
                continue

            base_idx = len(clipped_points)
            clipped_points.extend(clipped_polygon)
            for tri_idx in range(1, len(clipped_polygon) - 1):
                clipped_faces.append([base_idx, base_idx + tri_idx, base_idx + tri_idx + 1])

        if not clipped_faces:
            return None

        return _normalize_surface(
            _to_polydata(
                np.asarray(clipped_points, dtype=np.float32),
                np.asarray(clipped_faces, dtype=np.int64),
            )
        )

    def _normalize_line_mesh(mesh):
        if mesh is None:
            return None
        if hasattr(mesh, "clean"):
            mesh = mesh.clean()
        if getattr(mesh, "n_points", 0) == 0 or getattr(mesh, "n_cells", 0) == 0:
            return None
        return mesh

    def _axis_score(mesh):
        if mesh is None:
            return None
        return float(np.asarray(mesh.points)[:, axis_idx].mean())

    def _half_side_score(mesh, plane_coord_value):
        mesh = _normalize_surface(mesh)
        if mesh is None:
            return None

        try:
            sample_points = np.asarray(mesh.cell_centers().points)
        except Exception:
            sample_points = np.asarray(mesh.points)
        if sample_points.size == 0:
            return None

        signed_distance = sample_points[:, axis_idx] - float(plane_coord_value)
        side_tolerance = max(float(cut_band), 0.25)
        off_plane = np.abs(signed_distance) > side_tolerance
        if np.any(off_plane):
            signed_distance = signed_distance[off_plane]
        if signed_distance.size == 0:
            return None

        return float(np.median(signed_distance))

    def _make_cut_display_poly(mesh):
        mesh = _normalize_surface(mesh)
        if mesh is None:
            return None
        pattern_name = str(cut_pattern).lower()
        if pattern_name not in {'checker', 'hatch'}:
            return mesh

        other_axes = [idx for idx in range(3) if idx != axis_idx]
        points = np.asarray(mesh.points)
        span_u = float(np.ptp(points[:, other_axes[0]]))
        span_v = float(np.ptp(points[:, other_axes[1]]))
        min_span = max(min(span_u, span_v), 1.0)
        if pattern_name == 'checker':
            if float(cut_checker_size) > 0:
                checker_size = float(cut_checker_size)
            else:
                checker_size = max(min_span / 10.0, 1.5)
            checker_u = np.floor(points[:, other_axes[0]] / checker_size).astype(np.int64)
            checker_v = np.floor(points[:, other_axes[1]] / checker_size).astype(np.int64)
            checker_mask = ((checker_u + checker_v) % 2) == 0

            point_rgb = np.tile(
                np.asarray(cut_checker_light, dtype=np.float32),
                (points.shape[0], 1),
            )
            point_rgb[checker_mask] = np.asarray(cut_checker_dark, dtype=np.float32)
        else:
            if float(cut_hatch_spacing) > 0:
                hatch_spacing = float(cut_hatch_spacing)
            else:
                hatch_spacing = max(min_span / 12.0, 1.5)

            angle_rad = np.deg2rad(float(cut_hatch_angle_deg))
            stripe_coord = (
                points[:, other_axes[0]] * np.cos(angle_rad)
                + points[:, other_axes[1]] * np.sin(angle_rad)
            )
            stripe_coord = stripe_coord - float(stripe_coord.min())
            stripe_phase = np.mod(stripe_coord / hatch_spacing, 1.0)
            line_fraction = float(np.clip(cut_hatch_line_fraction, 0.05, 0.95))
            hatch_mask = stripe_phase < line_fraction

            point_rgb = np.tile(
                np.asarray(cut_hatch_light, dtype=np.float32),
                (points.shape[0], 1),
            )
            point_rgb[hatch_mask] = np.asarray(cut_hatch_dark, dtype=np.float32)

        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        return _to_polydata(points, faces, point_rgb=point_rgb)

    def _strip_plane_cap(mesh):
        mesh = _normalize_surface(mesh)
        if mesh is None or getattr(mesh, "n_cells", 0) == 0:
            return mesh

        mesh_normals = mesh.compute_normals(
            cell_normals=True,
            point_normals=False,
            inplace=False,
        )
        cell_centers = np.asarray(mesh.cell_centers().points)
        cell_normals = np.asarray(mesh_normals.cell_data["Normals"])
        cap_tolerance = max(float(cut_band), 0.25)
        cap_by_center = np.abs(cell_centers[:, axis_idx] - plane_coord) <= cap_tolerance
        cap_by_normal = np.abs(cell_normals[:, axis_idx]) >= 0.9
        cap_ids = np.flatnonzero(cap_by_center & cap_by_normal)
        if cap_ids.size == 0:
            return mesh

        shell_ids = np.setdiff1d(
            np.arange(mesh.n_cells, dtype=np.int64),
            cap_ids,
            assume_unique=False,
        )
        if shell_ids.size == 0:
            return mesh
        return _normalize_surface(mesh.extract_cells(shell_ids))

    def _extract_cut_outline(mesh):
        mesh = _normalize_surface(mesh)
        if mesh is None or not hasattr(mesh, "extract_feature_edges"):
            return None
        try:
            outline = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )
        except TypeError:
            outline = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
            )
        return _normalize_line_mesh(outline)

    def _build_cut_cap_poly(vol, plane_index):
        if plane_index <= 0 or plane_index >= vol.shape[axis_idx]:
            return None
        if axis_idx == 0:
            plane_field = 0.5 * (vol[plane_index - 1, :, :] + vol[plane_index, :, :])
        elif axis_idx == 1:
            plane_field = 0.5 * (vol[:, plane_index - 1, :] + vol[:, plane_index, :])
        else:
            plane_field = 0.5 * (vol[:, :, plane_index - 1] + vol[:, :, plane_index])

        if plane_field.shape[0] < 2 or plane_field.shape[1] < 2:
            return None

        upsample = max(int(cap_upsample), 1)
        coord_step = 1.0
        if upsample > 1:
            target_shape = (
                (plane_field.shape[0] - 1) * upsample + 1,
                (plane_field.shape[1] - 1) * upsample + 1,
            )
            plane_field = resize(
                plane_field,
                target_shape,
                order=1,
                mode='edge',
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.float32, copy=False)
            coord_step = 1.0 / float(upsample)

        cell_field = 0.25 * (
            plane_field[:-1, :-1]
            + plane_field[1:, :-1]
            + plane_field[:-1, 1:]
            + plane_field[1:, 1:]
        )
        filled = np.argwhere(cell_field >= float(iso_level))
        if filled.shape[0] == 0:
            return None

        other_axes = [idx for idx in range(3) if idx != axis_idx]
        points = np.zeros((filled.shape[0] * 4, 3), dtype=np.float32)
        faces = np.zeros((filled.shape[0], 5), dtype=np.int64)

        for cell_idx, (u_idx, v_idx) in enumerate(filled):
            base = 4 * cell_idx
            quad = np.zeros((4, 3), dtype=np.float32)
            quad[:, axis_idx] = plane_coord
            quad[0, other_axes[0]] = float(u_idx) * coord_step
            quad[0, other_axes[1]] = float(v_idx) * coord_step
            quad[1, other_axes[0]] = float(u_idx + 1) * coord_step
            quad[1, other_axes[1]] = float(v_idx) * coord_step
            quad[2, other_axes[0]] = float(u_idx + 1) * coord_step
            quad[2, other_axes[1]] = float(v_idx + 1) * coord_step
            quad[3, other_axes[0]] = float(u_idx) * coord_step
            quad[3, other_axes[1]] = float(v_idx + 1) * coord_step
            points[base:base + 4] = quad
            faces[cell_idx] = np.array([4, base, base + 1, base + 2, base + 3], dtype=np.int64)

        return _normalize_surface(pv.PolyData(points, faces.ravel()))

    solid_np = _to_numpy(solid_mask).astype(np.float32)
    if solid_np.ndim != 3:
        raise ValueError("solid_mask must be a 3D array shaped (D, H, W).")

    slice_axis = str(slice_axis).lower()
    axis_lookup = {'x': 0, 'y': 1, 'z': 2}
    if slice_axis not in axis_lookup:
        raise ValueError("slice_axis must be one of 'x', 'y', or 'z'.")
    axis_idx = axis_lookup[slice_axis]
    axis_len = solid_np.shape[axis_idx]
    if axis_len < 2:
        raise ValueError("Cutaway visualization requires at least 2 voxels along the slice axis.")

    if slice_idx is None:
        center_slice_idx = axis_len // 2
        offset_slices = int(np.round(float(slice_offset_frac) * float(axis_len)))
        slice_idx = center_slice_idx + offset_slices
    slice_idx = int(np.clip(slice_idx, 1, axis_len - 1))

    keep = str(keep).lower()
    if keep not in {'positive', 'negative'}:
        raise ValueError("keep must be 'positive' or 'negative'.")

    if smooth_sigma and smooth_sigma > 0:
        solid_vis = gaussian_filter(solid_np, sigma=float(smooth_sigma))
    else:
        solid_vis = solid_np

    full_verts, full_faces = _extract_mesh(solid_vis)
    if full_verts is None:
        raise ValueError(
            "Solid mask produced no visible surface. "
            "Check the solid mask convention or the chosen iso level."
        )
    full_poly = _normalize_surface(_to_polydata(full_verts, full_faces))
    if full_poly is None:
        raise ValueError("Failed to build a valid surface mesh from the solid mask.")

    plane_coord = float(slice_idx - 0.5)
    volume_size = np.maximum(np.asarray(solid_np.shape, dtype=np.float32) - 1.0, 1.0)
    volume_center = 0.5 * volume_size

    plane_center = np.asarray(volume_center, dtype=np.float32)
    plane_center[axis_idx] = plane_coord
    normal = np.zeros(3, dtype=np.float32)
    normal[axis_idx] = 1.0
    cap_poly = _build_cut_cap_poly(solid_vis, slice_idx)
    cap_display_poly = _make_cut_display_poly(cap_poly)

    negative_half = _clip_half(full_poly, plane_coord, 'negative')
    positive_half = _clip_half(full_poly, plane_coord, 'positive')
    if negative_half is None and positive_half is None:
        raise ValueError(
            "Cutaway produced no visible surface after clipping. "
            "Check the slice index and PyVista clipping support."
        )

    if keep == 'positive':
        cut_poly, removed_poly = positive_half, negative_half
    else:
        cut_poly, removed_poly = negative_half, positive_half

    if cut_poly is None:
        cut_poly = removed_poly
        removed_poly = None

    cut_faces = np.asarray(cut_poly.faces).reshape(-1, 4)[:, 1:]
    cut_points = np.asarray(cut_poly.points)
    cut_mask = np.abs(cut_points[:, axis_idx] - plane_coord) <= float(cut_band)
    cut_face_ids = np.flatnonzero(np.all(cut_mask[cut_faces], axis=1))
    cut_poly_normals = cut_poly.compute_normals(
        cell_normals=True,
        point_normals=False,
        inplace=False,
    )
    cell_centers = np.asarray(cut_poly.cell_centers().points)
    cell_normals = np.asarray(cut_poly_normals.cell_data["Normals"])
    cap_tolerance = max(float(cut_band), 0.25)
    cap_by_center = np.abs(cell_centers[:, axis_idx] - plane_coord) <= cap_tolerance
    cap_by_normal = np.abs(cell_normals[:, axis_idx]) >= 0.9
    cap_cell_ids = np.flatnonzero(cap_by_center & cap_by_normal)
    if cap_cell_ids.size > 0:
        cut_face_ids = np.union1d(cut_face_ids, cap_cell_ids).astype(np.int64, copy=False)
    shell_face_ids = np.setdiff1d(
        np.arange(cut_poly.n_cells, dtype=np.int64),
        cut_face_ids,
        assume_unique=False,
    )
    shell_poly = None
    if shell_face_ids.size > 0:
        shell_poly = _normalize_surface(cut_poly.extract_cells(shell_face_ids))
    cut_face_poly = None
    if cut_face_ids.size > 0:
        cut_face_poly = _normalize_surface(cut_poly.extract_cells(cut_face_ids))
    cut_face_display_poly = _make_cut_display_poly(cut_face_poly)
    cut_outline_source = cap_poly if cap_poly is not None else cut_face_poly
    cut_outline_poly = _extract_cut_outline(cut_outline_source) if show_cut_outline else None

    context_poly = full_poly if show_context else None
    if not show_removed_wire:
        removed_poly = None
    elif removed_poly is not None:
        removed_poly = _strip_plane_cap(removed_poly)

    center = np.asarray(volume_center, dtype=np.float32)
    size = np.asarray(volume_size, dtype=np.float32)
    diag = float(np.linalg.norm(size))
    if diag <= 0:
        diag = float(np.sqrt(np.sum(np.square(solid_np.shape))))

    off_screen = save_path is not None
    pl = pv.Plotter(off_screen=off_screen, window_size=window_size)
    pl.set_background(background)

    if context_poly is not None:
        pl.add_mesh(
            context_poly,
            color=outer_color,
            opacity=context_opacity,
            smooth_shading=True,
        )

    if removed_poly is not None:
        pl.add_mesh(
            removed_poly,
            color=removed_wire_color,
            opacity=removed_wire_opacity,
            style='wireframe',
            line_width=removed_wire_width,
            lighting=False,
        )

    if show_cut_plane:
        other_axes = [idx for idx in range(3) if idx != axis_idx]
        plane = pv.Plane(
            center=plane_center,
            direction=normal,
            i_size=max(float(size[other_axes[0]]) * float(cut_plane_scale), 1.0),
            j_size=max(float(size[other_axes[1]]) * float(cut_plane_scale), 1.0),
            i_resolution=1,
            j_resolution=1,
        )
        pl.add_mesh(
            plane,
            color=cut_plane_color,
            opacity=cut_plane_opacity,
            lighting=False,
        )

    if shell_poly is not None:
        pl.add_mesh(
            shell_poly,
            color=outer_color,
            smooth_shading=True,
            opacity=1.0,
        )

    if cap_display_poly is not None:
        if str(cut_pattern).lower() in {'checker', 'hatch'}:
            pl.add_mesh(
                cap_display_poly,
                scalars="colors",
                rgb=True,
                opacity=1.0,
                lighting=False,
            )
        else:
            pl.add_mesh(
                cap_display_poly,
                color=cut_color,
                opacity=1.0,
                lighting=False,
            )
    elif cut_face_display_poly is not None:
        if str(cut_pattern).lower() in {'checker', 'hatch'}:
            pl.add_mesh(
                cut_face_display_poly,
                scalars="colors",
                rgb=True,
                opacity=1.0,
                lighting=False,
            )
        else:
            pl.add_mesh(
                cut_face_display_poly,
                color=cut_color,
                opacity=1.0,
                lighting=False,
            )

    if cut_outline_poly is not None:
        pl.add_mesh(
            cut_outline_poly,
            color=cut_outline_color,
            opacity=cut_outline_opacity,
            line_width=cut_outline_width,
            lighting=False,
        )

    elev_deg, azim_deg = float(view_angle[0]), float(view_angle[1])
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    cam_pos = center + float(camera_distance_scale) * diag * np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
        dtype=np.float32,
    )
    pl.camera_position = [cam_pos.tolist(), center.tolist(), [0, 0, 1]]
    pl.camera.zoom(1.1)

    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
        pl.show(screenshot=str(save_path), auto_close=True)
    else:
        pl.show()



def compute_com(solid_mask):
    # solid_mask: torch.Tensor (D, H, W)
    mask = solid_mask.detach().cpu().numpy()
    total = mask.sum()
    if total == 0:
        return np.array([0, 0, 0])
    Z, Y, X = np.meshgrid(
        np.arange(mask.shape[0]),
        np.arange(mask.shape[1]),
        np.arange(mask.shape[2]),
        indexing='ij'
    )
    com_z = (Z * mask).sum() / total
    com_y = (Y * mask).sum() / total
    com_x = (X * mask).sum() / total
    return np.array([com_z, com_y, com_x])  # (x, y, z) for Vispy



def visualize_3d_solid_gray_vispy_overlay(
    solid_mask_list,
    iso_level=0.5,
    step_size=None,
    to_consider=None,
    save_path=None,
    view_angle=(30, 45),        # (elevation, azimuth)
    depth_axis='y',             # 'x' | 'y' | 'z' used for grayscale mapping
    mesh_opacity=1.0,           # 0..1
    camera_distance=None        # if None, auto-fit
):
    """
    Visualize a 3D solid (only) in grayscale on a white background using VisPy.

    Notes:
    - Robust against frames with no surface at the given iso level (mesh hidden for those frames).
    - Recomputes vertex colors each frame so colors always match the current vertex count.
    - No rotate-camera or axis options (per your request).

    Args:
        solid_mask_list: list of (D, H, W) volumes (torch.Tensor or np.ndarray; float/bool) per timestep.
        iso_level: isosurface level for marching cubes.
        fps: frames per second for animation/video.
        save_path: optional output video path (e.g., 'out.mp4' or 'out.gif'). If None, runs interactively.
        view_angle: (elevation, azimuth) camera angles in degrees.
        save_frames: if int, also save every Nth frame as PNG beside the video.
        depth_axis: which axis to map to grayscale for depth cue ('x', 'y', or 'z').
        mesh_opacity: RGBA alpha for the mesh [0..1].
        camera_distance: override camera distance; if None, chosen from volume diagonal.
    """
    assert len(solid_mask_list) > 0, "solid_mask_list must be non-empty."

    def _to_numpy(vol):
        if torch is not None and isinstance(vol, torch.Tensor):
            return vol.detach().cpu().numpy()
        return np.asarray(vol)

    # Use (D, H, W) volumes directly; marching_cubes expects (z, y, x) which matches (D, H, W).
    # vol0 = _to_numpy(solid_mask_list[0])
    if to_consider is not None:
        vols_to_consider = [solid_mask_list[i] for i in to_consider]
        first_com = to_consider[0]
        last_com = to_consider[-1]
    elif step_size is not None:
        vols_to_consider = [vol for vol in solid_mask_list[::step_size]]
        first_com = 0
        last_com = len(solid_mask_list) - len(solid_mask_list) % step_size - 1
    else:
        vols_to_consider = solid_mask_list
        first_com = 0
        last_com = len(solid_mask_list) - 1
    vol0 = _to_numpy(torch.min(torch.stack(vols_to_consider), dim=0).values)
    assert vol0.ndim == 3, "Each solid mask must be 3D (D, H, W)."
    D, H, W = vol0.shape

    # Canvas & camera
    # Set canvas size based on H, W (scale up for visibility, but cap at 1200x1200)
    scale = 10  # pixels per voxel (adjust as needed)
    width = min(int(W * scale), 12000)
    height = min(int(H * scale), 12000)
    canvas = scene.SceneCanvas(
        keys='interactive',
        show=(save_path is None),
        bgcolor='white',
        size=(height, width),
        dpi=120,
    )
    view = canvas.central_widget.add_view()

    if camera_distance is None:
        diag = float(np.sqrt(D**2 + H**2 + W**2))
        camera_distance = 0.8 * diag

    cam = scene.cameras.TurntableCamera(
        azimuth=float(view_angle[1]),
        elevation=float(view_angle[0]),
        distance=float(camera_distance),
        center=(W/2.0, H/2.0, D/2.0)
    )
    view.camera = cam

    # Choose axis index for grayscale mapping after converting verts to (x,y,z)
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    depth_idx = axis_map.get(str(depth_axis).lower(), 0)  # default 'z'

    mesh_visual = None

    coms = np.stack([compute_com(1 - solid_mask) for solid_mask in solid_mask_list])  # shape (T, 3)
    # print(coms)
    com_line = scene.visuals.Line(
        pos=coms[first_com:last_com], color=(0, 0, 1, 0.5), width=4, parent=view.scene, method='gl'
    )
    com_line.set_gl_state(depth_test=True, blend=True)  # Enable depth test

    def _make_mesh(volDHW):
        """
        Returns (verts_xyz, faces, colors_rgba) or (None, None, None) if no surface.
        - verts_xyz: float32 (N, 3) in (x, y, z) for VisPy
        - faces: uint32 (M, 3)
        - colors_rgba: float32 (N, 4), per-vertex grayscale in [0,1]
        """
        vmin = float(volDHW.min())
        vmax = float(volDHW.max())

        # marching_cubes requires iso strictly inside (vmin, vmax)
        if not (vmin < iso_level < vmax):
            return None, None, None

        # Extract surface. marching_cubes returns verts in (z, y, x) ~ (D, H, W)
        verts_zyx, faces, _, _ = marching_cubes(volDHW, level=iso_level)

        # Convert to VisPy (x, y, z)
        verts_xyz = verts_zyx[:, [0, 1, 2]].astype(np.float32)
        faces = faces.astype(np.uint32)

        # Per-vertex grayscale (no depth shading; use fixed gray)
        g = np.full(verts_xyz.shape[0], 0.5, dtype=np.float32)
        # coord = verts_xyz[:, depth_idx]
        # cmin, cmax = float(coord.min()), float(coord.max())
        # if cmax > cmin:
        #     g = (coord - cmin) / (cmax - cmin)
        # else:
        #     g = np.zeros_like(coord, dtype=np.float32)

        # Keep some contrast vs white background
        g = 0.15 + 0.75 * g  # -> [0.15, 0.90]
        g = g.astype(np.float32)

        colors = np.empty((verts_xyz.shape[0], 4), dtype=np.float32)
        colors[:, 0] = g  # R
        colors[:, 1] = g  # G
        colors[:, 2] = g  # B
        colors[:, 3] = np.float32(mesh_opacity)  # A

        return verts_xyz, faces, colors

    def _update_mesh(volDHW):
        nonlocal mesh_visual
        verts, faces, colors = _make_mesh(volDHW)

        if verts is None:
            # No surface this frame
            if mesh_visual is not None:
                mesh_visual.visible = False
            return

        if mesh_visual is None:
            mesh_visual = scene.visuals.Mesh(
                vertices=verts,
                faces=faces,
                vertex_colors=colors,   # per-vertex RGBA (float32 0..1)
                shading='smooth',
                parent=view.scene,
            )
            # Depth test ON; blend only if opacity < 1
            mesh_visual.set_gl_state(depth_test=False, blend=(mesh_opacity < 1.0), cull_face=True)
            mesh_visual.visible = True
        else:
            # IMPORTANT: always pass a fresh color array that matches the *current* vertex count
            mesh_visual.set_data(vertices=verts, faces=faces, vertex_colors=colors)
            mesh_visual.visible = True

    # Initial draw
    _update_mesh(vol0)



    T = len(solid_mask_list)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # writer = imageio.get_writer(save_path, fps=fps)

        # Update COM line
        # com_line.set_data(pos=coms[:t+1])

        # lock camera to requested angle (no rotation per your request)
        cam.azimuth = float(view_angle[1])
        cam.elevation = float(view_angle[0])

        frame = np.asarray(canvas.render())
        # writer.append_data(frame)

        imageio.imwrite(save_path, frame)
        # writer.close()


def visualize_solid_mask_cutaway(
    solid_mask,
    *,
    slice_axis='x',
    slice_idx=None,
    keep='positive',
    iso_level=0.5,
    save_path=None,
    view_angle=(20, 30),
    smooth_sigma=0.6,
    cut_band=0.75,
    show_context=False,
    context_alpha=0.08,
    outer_color=(0.62, 0.64, 0.68, 1.0),
    cut_color=(0.88, 0.42, 0.22, 1.0),
    background='white',
    camera_distance=None,
    canvas_size=1000,
):
    """
    Render a 3D cutaway view of a voxel solid by clipping one half away.

    The remaining half is shown as a shaded mesh, and vertices near the exposed
    clipping plane are colored separately so the cross-section reads clearly.
    """
    def _to_numpy(vol):
        if torch is not None and isinstance(vol, torch.Tensor):
            return vol.detach().cpu().numpy()
        return np.asarray(vol)

    def _extract_mesh(vol, vertex_color, plane_coord=None, plane_color=None):
        vmin = float(vol.min())
        vmax = float(vol.max())
        if not (vmin < iso_level < vmax):
            return None, None, None

        verts, faces, _, _ = marching_cubes(vol, level=iso_level)
        verts = verts.astype(np.float32)
        faces = faces.astype(np.uint32)
        colors = np.tile(np.asarray(vertex_color, dtype=np.float32), (verts.shape[0], 1))

        if plane_coord is not None and plane_color is not None:
            on_plane = np.abs(verts[:, axis_idx] - plane_coord) <= float(cut_band)
            colors[on_plane] = np.asarray(plane_color, dtype=np.float32)

        return verts, faces, colors

    solid_np = _to_numpy(solid_mask).astype(np.float32)
    if solid_np.ndim != 3:
        raise ValueError("solid_mask must be a 3D array shaped (D, H, W).")

    D, H, W = solid_np.shape
    slice_axis = str(slice_axis).lower()
    axis_lookup = {'x': 0, 'y': 1, 'z': 2}
    if slice_axis not in axis_lookup:
        raise ValueError("slice_axis must be one of 'x', 'y', or 'z'.")
    axis_idx = axis_lookup[slice_axis]
    axis_len = solid_np.shape[axis_idx]
    if axis_len < 2:
        raise ValueError("Cutaway visualization requires at least 2 voxels along the slice axis.")

    if slice_idx is None:
        slice_idx = axis_len // 2
    slice_idx = int(np.clip(slice_idx, 1, axis_len - 1))

    keep = str(keep).lower()
    if keep not in {'positive', 'negative'}:
        raise ValueError("keep must be 'positive' or 'negative'.")

    clipped_np = solid_np.copy()
    cut_slices = [slice(None)] * 3
    if keep == 'positive':
        cut_slices[axis_idx] = slice(0, slice_idx)
    else:
        cut_slices[axis_idx] = slice(slice_idx, None)
    clipped_np[tuple(cut_slices)] = 0.0

    if smooth_sigma and smooth_sigma > 0:
        solid_vis = gaussian_filter(solid_np, sigma=float(smooth_sigma))
        clipped_vis = gaussian_filter(clipped_np, sigma=float(smooth_sigma))
    else:
        solid_vis = solid_np
        clipped_vis = clipped_np

    plane_coord = float(slice_idx - 0.5)
    full_verts = full_faces = full_colors = None
    if show_context:
        context_rgba = (
            float(outer_color[0]),
            float(outer_color[1]),
            float(outer_color[2]),
            float(context_alpha),
        )
        full_verts, full_faces, full_colors = _extract_mesh(solid_vis, context_rgba)
    cut_verts, cut_faces, cut_colors = _extract_mesh(
        clipped_vis,
        outer_color,
        plane_coord=plane_coord,
        plane_color=cut_color,
    )

    if cut_verts is None:
        raise ValueError(
            "Cutaway produced no visible surface. "
            "Check the solid mask convention or choose a different slice index."
        )

    bbox_min = cut_verts.min(axis=0)
    bbox_max = cut_verts.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 0:
        diag = float(np.sqrt(D ** 2 + H ** 2 + W ** 2))

    canvas = scene.SceneCanvas(
        keys='interactive',
        show=(save_path is None),
        bgcolor=background,
        size=(int(canvas_size), int(canvas_size)),
        dpi=120,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        azimuth=float(view_angle[1]),
        elevation=float(view_angle[0]),
        distance=float(camera_distance) if camera_distance is not None else 1.55 * diag,
        center=tuple(center.tolist()),
        fov=45.0,
    )

    if show_context and full_verts is not None:
        context_mesh = scene.visuals.Mesh(
            vertices=full_verts,
            faces=full_faces,
            vertex_colors=full_colors,
            shading='smooth',
            parent=view.scene,
        )
        context_mesh.set_gl_state(depth_test=True, blend=True, cull_face=True)

    cut_mesh = scene.visuals.Mesh(
        vertices=cut_verts,
        faces=cut_faces,
        vertex_colors=cut_colors,
        shading='smooth',
        parent=view.scene,
    )
    cut_mesh.set_gl_state(depth_test=True, blend=True, cull_face=True)

    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
        imageio.imwrite(str(save_path), np.asarray(canvas.render()))
    else:
        app.run()


def visualize_slice(rho, u, step=None, save=False, filename="frame.png"):
    """
    Visualizes a 2D slice from 3D scalar and velocity field.
    """
    D, H, W = rho.shape
    z = D // 2  # Take middle Z slice

    slice_rho = rho[z].detach().cpu().numpy()
    slice_u = u[z].detach().cpu().numpy()  # shape [H, W, 3]

    # Velocity magnitude
    mag = np.linalg.norm(slice_u, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(slice_rho, origin='lower', cmap='plasma')
    axes[0].set_title("Density slice (Z={})".format(z))
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mag, origin='lower', cmap='viridis')
    axes[1].set_title("Velocity magnitude")
    fig.colorbar(im1, ax=axes[1])

    if save:
        out = f"{filename}" if step is None else f"{filename}_{step:04d}.png"
        plt.savefig(out)
    # plt.show()
    plt.close()


def visualize_solid_mask_slice(
    solid_mask, 
    slice_axis='x', 
    slice_idx=None, 
    level=0.5, 
    interactive=False, 
    save=False, 
    filename="solid_mask_slice.png",
    clean_plot=False
):
    """
    Visualize 2D slice of solid mask along one axis, with level set contour.

    Args:
        solid_mask: tensor (D, H, W)
        slice_axis: 'z', 'y', or 'x'
        slice_idx: index of slice (default: center slice)
        level: level set threshold (default 0.5)
        interactive: if True, allow interactive editing
        save: if True, save the figure to file
        filename: file name for saving
    """
    solid_np = solid_mask.cpu().numpy()
    D, H, W = solid_np.shape

    # ver large text
    plt.rcParams.update({'font.size': 16})

    if slice_axis == 'x':
        if slice_idx is None:
            slice_idx = D // 2
        slice_2d = solid_np[slice_idx, :, :]
        xlabel, ylabel = 'X', 'Y'
    elif slice_axis == 'y':
        if slice_idx is None:
            slice_idx = H // 2
        slice_2d = solid_np[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    elif slice_axis == 'z':
        if slice_idx is None:
            slice_idx = W // 2
        slice_2d = solid_np[:, :, slice_idx]
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")

    if interactive:
        one = False
        while True:
            one = not one
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_2d.T, origin='lower', cmap='viridis')
            plt.colorbar()
            plt.title(f"Click to add {one} (press enter when done)")
            plt.contour(slice_2d.T, levels=[level], colors='red', linewidths=1.5)
            plt.show(block=False)

            # User clicks
            points = plt.ginput(n=-1, timeout=0)
            plt.close()

            # Apply edits
            for x, y in points:
                xi = int(x)
                yi = int(y)
                slice_2d[xi, yi] = 1.0 if one else 0.0
    # Non-interactive or after editing: just show or save the slice
    plt.figure(figsize=(6, 6))
    plt.imshow(1-slice_2d.T, origin='lower', cmap='Greys')
    if not clean_plot:
        plt.colorbar(label='Solid mask')
        plt.title(f"Solid mask slice at {slice_axis}={slice_idx}")
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.contour(slice_2d.T, levels=[level], colors='red', linewidths=1.5)
    # tight layout
    plt.tight_layout()
    # get rid of extra whitespace

    # no axis ticks
    plt.axis('off')
    if save:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()
