import json
from matplotlib import pyplot as plt
import torch
import numpy as np
from datetime import datetime
import sys, os
from pathlib import Path
import gc
import argparse
import copy

RELEASE_ROOT = Path(__file__).resolve().parent
for relative_path in ("code/torch-splatting", "code/eulerian-fluid"):
    package_root = str(RELEASE_ROOT / relative_path)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)

import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer

from eulerian_fluid.euler_checkpoint import simulate_physics_loss


USE_GPU_PYTORCH = True

NUM_GAUSSIANS = 2**14


def sanitize_arguments(args):
    sanitized_args = {}
    for k, v in args.items():
        if isinstance(v, torch.Tensor):
            sanitized_args[k] = v.detach().cpu()
        else:
            sanitized_args[k] = v
    return sanitized_args


def merge_dicts(default, custom):
    for k, v in custom.items():
        if k in default and isinstance(default[k], dict) and isinstance(v, dict):
            merge_dicts(default[k], v)
        else:
            default[k] = v


def parse_override_value(raw_value):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def parse_override_path(path):
    if not path:
        raise ValueError("Override path cannot be empty")

    tokens = []
    i = 0
    while i < len(path):
        if path[i] == '.':
            raise ValueError(f"Invalid override path '{path}': empty path segment")

        key_start = i
        while i < len(path) and path[i] not in '.[':
            i += 1

        if key_start == i:
            raise ValueError(f"Invalid override path '{path}': expected a key at position {i}")

        tokens.append(path[key_start:i])

        while i < len(path) and path[i] == '[':
            i += 1
            index_start = i
            while i < len(path) and path[i] != ']':
                i += 1

            if i >= len(path):
                raise ValueError(f"Invalid override path '{path}': missing closing ']'")

            index_text = path[index_start:i]
            if not index_text.isdigit():
                raise ValueError(
                    f"Invalid override path '{path}': list index '{index_text}' must be a non-negative integer"
                )

            tokens.append(int(index_text))
            i += 1

        if i < len(path):
            if path[i] != '.':
                raise ValueError(
                    f"Invalid override path '{path}': unexpected character '{path[i]}' at position {i}"
                )
            i += 1
            if i >= len(path):
                raise ValueError(f"Invalid override path '{path}': cannot end with '.'")

    return tokens


def format_override_path(tokens):
    parts = []
    for token in tokens:
        if isinstance(token, int):
            if not parts:
                raise ValueError("Override path cannot start with a list index")
            parts[-1] = f"{parts[-1]}[{token}]"
        else:
            parts.append(token)
    return '.'.join(parts)


def apply_override(config, path_tokens, value):
    if not path_tokens:
        raise ValueError("Override path cannot be empty")

    current = config
    traversed = []

    for i, token in enumerate(path_tokens[:-1]):
        next_token = path_tokens[i + 1]
        current_path = format_override_path(traversed) if traversed else "<root>"

        if isinstance(token, str):
            if not isinstance(current, dict):
                raise TypeError(
                    f"Cannot access key '{token}' on non-dict object at '{current_path}'"
                )

            if token not in current:
                if isinstance(next_token, int):
                    missing_path = format_override_path(traversed + [token])
                    raise KeyError(
                        f"Cannot create missing list '{missing_path}' via indexed override; replace the whole list instead"
                    )
                current[token] = {}

            current = current[token]
            traversed.append(token)
            continue

        if not isinstance(current, list):
            raise TypeError(f"Cannot index into non-list object at '{current_path}'")
        if token >= len(current):
            raise IndexError(f"List index {token} out of range at '{current_path}'")

        current = current[token]
        traversed.append(token)

    final_token = path_tokens[-1]
    current_path = format_override_path(traversed) if traversed else "<root>"

    if isinstance(final_token, str):
        if not isinstance(current, dict):
            raise TypeError(
                f"Cannot set key '{final_token}' on non-dict object at '{current_path}'"
            )
        current[final_token] = value
        return

    if not isinstance(current, list):
        raise TypeError(f"Cannot index into non-list object at '{current_path}'")
    if final_token >= len(current):
        raise IndexError(f"List index {final_token} out of range at '{current_path}'")
    current[final_token] = value


def parse_override_assignment(assignment):
    if '=' not in assignment:
        raise ValueError(
            f"Invalid override '{assignment}': expected PATH=VALUE syntax"
        )

    path_text, raw_value = assignment.split('=', 1)
    path_text = path_text.strip()
    path_tokens = parse_override_path(path_text)
    return format_override_path(path_tokens), path_tokens, parse_override_value(raw_value)


def create_run_results_folder(results_folder, experiment_name, notes, experiment_settings=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(results_folder) / experiment_name / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'notes.txt', 'a') as f:
        f.write(notes)
    if experiment_settings is None:
        experiment_settings = notes
    with open(run_dir / 'experiment_settings.json', 'w') as f:
        if isinstance(experiment_settings, str):
            f.write(experiment_settings)
        else:
            json.dump(experiment_settings, f, indent=2)
    return run_dir


def load_experiment_settings_from_checkpoint(profile_checkpoint):
    checkpoint_path = Path(profile_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'experiment_settings' in checkpoint:
        return copy.deepcopy(checkpoint['experiment_settings'])

    run_dir = checkpoint_path.resolve().parent.parent
    settings_json = run_dir / 'experiment_settings.json'
    if settings_json.exists():
        with open(settings_json, 'r') as f:
            return json.load(f)

    notes_path = run_dir / 'notes.txt'
    if notes_path.exists():
        with open(notes_path, 'r') as f:
            notes_text = f.read().strip()
        try:
            return json.loads(notes_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse experiment settings from {notes_path}. "
                "Expected JSON notes or a checkpoint that embeds experiment_settings."
            ) from exc

    raise FileNotFoundError(
        f"Could not find experiment settings near checkpoint {checkpoint_path}. "
        "Expected embedded experiment_settings, experiment_settings.json, or notes.txt."
    )


def format_dir_scalar(value):
    text = f"{float(value):.3f}".rstrip('0').rstrip('.')
    if text == "-0":
        text = "0"
    return text.replace('-', 'm').replace('.', 'p')


def format_dir_vector(prefix, values):
    return f"{prefix}_{'_'.join(format_dir_scalar(v) for v in values)}"


def build_rotation_setting_dirname(index, setting):
    parts = [
        f"physics_loss_case_{index:02d}",
        f"w_{format_dir_scalar(setting['weight'])}",
        format_dir_vector("pos", setting['init_solid_pos']),
        format_dir_vector("q", setting['init_solid_q']),
        format_dir_vector("v", setting['init_solid_v']),
        format_dir_vector("omega", setting['init_solid_omega']),
    ]
    return "__".join(parts)


def normalize_experiment_settings(experiment_settings):
    physics_settings = experiment_settings.get('physics', {})
    physics_settings.pop('fit_fraction', None)
    physics_settings.pop('cosine_anneal', None)
    physics_settings.pop('exp_anneal', None)
    physics_settings.pop('mahalanobis_distance_threshold', None)
    physics_settings.pop('threshold_sharpness', None)

    for rotation_setting in physics_settings.get('rotation_loss_scalings', []):
        rotation_setting.pop('axis', None)
        rotation_setting.pop('angle_radians', None)

    experiment_settings.pop('shape_regularization', None)

    opacity_settings = experiment_settings.get('opacity_loss', {})
    opacity_loss_settings = opacity_settings.get('loss_settings', {})
    opacity_loss_settings.pop('target_opacity', None)

    return experiment_settings


def build_physics_sim_args(
    experiment_settings,
    *,
    step,
    save_dir,
    rotation_setting,
    gaussian_means=None,
    gaussian_scalings=None,
    gaussian_rotations=None,
    gaussian_opacities=None,
):
    physics_settings = experiment_settings['physics']
    brinkman_lambda = physics_settings.get('brinkman_lambda', 300.0)
    return {
        'gaussian_means': gaussian_means,
        'gaussian_scalings': gaussian_scalings,
        'gaussian_rotations': gaussian_rotations,
        'gaussian_opacities': gaussian_opacities,
        'step': step,
        'save_dir': save_dir,
        'disable_checkpoint': physics_settings.get('disable_checkpoint', False),
        'checkpoint_interval': physics_settings.get('checkpoint_interval', 10),
        'tile_size': physics_settings.get('tile_size', 8),
        'do_visualize': physics_settings.get('do_visualize', True),
        'mask_sharpening': physics_settings.get('mask_sharpening', 0.05),
        'loss_specs': copy.deepcopy(physics_settings.get('loss_specs', [])),
        'init_solid_pos': rotation_setting['init_solid_pos'],
        'init_solid_q': rotation_setting['init_solid_q'],
        'init_solid_v': rotation_setting['init_solid_v'],
        'init_solid_omega': rotation_setting['init_solid_omega'],
        'move_object': physics_settings.get('move_object', False),
        'voxels_per_unit': physics_settings.get('voxels_per_unit', 32.0),
        'num_pressure_iterations': physics_settings.get('num_pressure_iterations', 10),
        'grid_shape': physics_settings['grid_shape'],
        'dt': physics_settings.get('dt', 0.15),
        'num_steps': physics_settings.get('num_steps', 30),
        'com_blob_radius': physics_settings.get('com_blob_radius', 0.0),
        'brinkman_lambda': brinkman_lambda,
    }


class GSSTrainer(Trainer):
    def __init__(self, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}), device=device)
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        self.physics_losses_log = []
        self.psnr_log = []
        self.ssim_log = []
        self.l1_log = []
        self.experiment_settings = kwargs.get('experiment_settings', {})
        self.physics_step_count = 0
        self.device = device
    
    def on_train_step(self):
        visual_loss_interval = self.experiment_settings['visual']['interval']

        l1_loss = torch.tensor(0.0, device=device)
        depth_loss = torch.tensor(0.0, device=device)
        ssim_loss = torch.tensor(0.0, device=device)
        psnr = torch.tensor(0.0, device=device)

        do_visual_loss = self.step % visual_loss_interval == 0 and self.experiment_settings['visual']['enabled']
        if do_visual_loss:
            ind = np.random.choice(len(self.data['camera']))
            camera = self.data['camera'][ind]
            rgb = self.data['rgb'][ind]
            depth = self.data['depth'][ind]
            mask = (self.data['alpha'][ind] > 0.5)
            if USE_GPU_PYTORCH:
                camera = to_viewpoint_camera(camera)

            out = self.gaussRender(pc=self.model, camera=camera)
            
            l1_loss = loss_utils.l1_loss(out['render'], rgb)
            depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
            ssim_loss = 1.0 - loss_utils.ssim(out['render'], rgb)
            
            psnr = utils.img2psnr(out['render'], rgb)

            self.psnr_log.append(psnr.item())
            self.ssim_log.append(1.0 - ssim_loss.item())
            self.l1_log.append(l1_loss.item())


        physics_loss_interval = self.experiment_settings['physics']['loss_interval']
        start_physics_loss = self.experiment_settings['physics']['start_step']

        do_physics_loss = self.step % physics_loss_interval == 0 and self.step >= start_physics_loss
        do_physics_loss = do_physics_loss and self.experiment_settings['physics']['run_simulation']
        physics_step_idx = self.physics_step_count if do_physics_loss else None
        physics_loss = torch.tensor(0.0, device=device)
        if do_physics_loss:
            means = self.model.get_xyz
            opacities = self.model.get_opacity
            scaling = self.model.get_scaling
            rotation = self.model.get_rotation

            for setting_idx, setting in enumerate(self.experiment_settings['physics']['rotation_loss_scalings']):
                weight = setting['weight']
                save_dir = self.results_folder / build_rotation_setting_dirname(setting_idx, setting)
                physics_sim_args = build_physics_sim_args(
                    self.experiment_settings,
                    step=self.step,
                    save_dir=save_dir,
                    rotation_setting=setting,
                    gaussian_means=means,
                    gaussian_scalings=scaling,
                    gaussian_rotations=rotation,
                    gaussian_opacities=opacities,
                )

                if self.step % 50 == 0:
                    os.makedirs(physics_sim_args['save_dir'] / 'args_checkpoints', exist_ok=True)
                    torch.save(sanitize_arguments(physics_sim_args), physics_sim_args['save_dir'] / 'args_checkpoints' / f'args_checkpoint_{self.step:04d}.pt')

                physics_loss += weight * simulate_physics_loss(**physics_sim_args)

            self.physics_losses_log.append(physics_loss.item())
            self.physics_step_count += 1


        if not self.experiment_settings['physics']['optimize_loss']:
            physics_loss = torch.tensor(0.0, device=device)


        do_blob_size_loss = self.experiment_settings['blob_size_loss']['enabled']
        blob_size_loss = torch.tensor(0.0, device=device)
        if do_blob_size_loss:
            scaling = self.model.get_scaling
            blob_size_loss = loss_utils.scaling_length_penalty(scaling, **self.experiment_settings['blob_size_loss']['loss_settings'])

        physics_lambda = self.experiment_settings['physics']['loss_scaling']
        blob_size_loss_lambda = self.experiment_settings['blob_size_loss']['weight']

        taper_physics_loss = self.experiment_settings['physics'].get('taper_physics_loss', False)
        if taper_physics_loss and do_physics_loss:
            taper_steps = 300
            taper_progress = min(physics_step_idx / taper_steps, 1.0)
            physics_lambda = physics_lambda * max(0.1, 1.0 - 0.9 * taper_progress)

        opacity_loss = torch.tensor(0.0, device=device)
        do_opacity_loss = self.experiment_settings['opacity_loss']['enabled']
        if do_opacity_loss:
            opacities = self.model.get_opacity
            epsilon = 1e-4
            opacity_loss = -torch.log(opacities + epsilon)
            if self.experiment_settings['opacity_loss']['loss_settings']['reduction'] == 'mean':
                opacity_loss = torch.mean(opacity_loss)
            elif self.experiment_settings['opacity_loss']['loss_settings']['reduction'] == 'sum':
                opacity_loss = torch.sum(opacity_loss)
        opacity_loss_lambda = self.experiment_settings['opacity_loss']['weight']

        if not do_physics_loss:
            total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth + physics_lambda * physics_loss + blob_size_loss_lambda * blob_size_loss + opacity_loss * opacity_loss_lambda
            log_dict = {'total': total_loss}
        else:
            total_loss = physics_lambda * physics_loss
            log_dict = {'total': total_loss, 'physics_loss': physics_loss, 'ssim_loss': ssim_loss, 'l1_loss': l1_loss}
            with open(self.results_folder / 'log.txt', 'a') as f:
                log_dict_ser = {}
                for k, v in log_dict.items():
                    if isinstance(v, torch.Tensor):
                        log_dict_ser[k] = v.item()
                    else:
                        log_dict_ser[k] = v
                f.write(json.dumps({**log_dict_ser, 'step': self.step}) + '\n')
        
        log_dict = {'total_loss': total_loss}

        return total_loss, log_dict
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint['step']
        print(f"Loaded checkpoint from {checkpoint_path} at step {self.step}")

    def on_evaluate_step(self, **kwargs):
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.results_folder / 'checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': self.step,
            'experiment_settings': copy.deepcopy(self.experiment_settings),
            'config': {
                "max_sh_degree": self.model.max_sh_degree,
            },
        }, self.results_folder / 'checkpoints' / f'checkpoint-{self.step}.pth')
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()

        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        os.makedirs(self.results_folder / 'images', exist_ok=True)
        utils.imwrite(str(self.results_folder / 'images' / f'image-{self.step}.png'), image)

        plt.plot(self.physics_losses_log)
        np.save(self.results_folder / 'physics_loss.npy', np.array(self.physics_losses_log))
        plt.savefig(self.results_folder / 'physics_loss.png')
        plt.close()

        plt.plot(self.psnr_log)
        np.save(self.results_folder / 'psnr_log.npy', np.array(self.psnr_log))
        plt.savefig(self.results_folder / 'psnr_log.png')
        plt.close()

        plt.plot(self.ssim_log)
        np.save(self.results_folder / 'ssim_log.npy', np.array(self.ssim_log))
        plt.savefig(self.results_folder / 'ssim_log.png')
        plt.close()

        plt.plot(self.l1_log)
        np.save(self.results_folder / 'l1_log.npy', np.array(self.l1_log))
        plt.savefig(self.results_folder / 'l1_log.png')
        plt.close()

def do_experiment(data_folder, results_folder, experiment_name, notes, experiment_settings, checkpoint=None):

    folder = data_folder
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)


    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'], num_layers=experiment_settings.get('num_layers', 1))
    raw_points = points.random_sample(NUM_GAUSSIANS)

    gaussModel = GaussModel(sh_degree=4, debug=False)
    gaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': True,
    }

    results_folder = create_run_results_folder(
        results_folder,
        experiment_name,
        notes,
        experiment_settings=experiment_settings,
    )

    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=experiment_settings['total_steps'],
        i_image =25,
        train_lr=0.5e-3, 
        amp=False,
        fp16=True,
        results_folder=results_folder,
        render_kwargs=render_kwargs,
        device=device,
        experiment_settings=experiment_settings,
    )

    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    trainer.on_evaluate_step()
    trainer.train()


def do_renders(checkpoint, results_folder, data_folder, device='cuda', output_dir_name='all_views'):
    output_folder = Path(results_folder) / output_dir_name
    output_folder.mkdir(parents=True, exist_ok=True)

    checkpoint_data = torch.load(checkpoint, map_location=device)

    gaussModel = GaussModel(sh_degree=checkpoint_data['config']['max_sh_degree'], debug=False)


    data = read_all(data_folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}

    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)


    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    raw_points = points.random_sample(NUM_GAUSSIANS)

    gaussModel.create_from_pcd(pcd=raw_points)

    gaussModel.load_state_dict(checkpoint_data['model_state_dict'])

    renderer = GaussRenderer(white_bkgd=True, device=device)

    for idx, camera in enumerate(data['camera']):
        cam = to_viewpoint_camera(camera)
        out = renderer(pc=gaussModel, camera=cam)
        out_vis = renderer.gaussian_visualize(
            pc=gaussModel,
            camera=cam,
            outline_thickness=0.03,
            outline_alpha=4.0,
            outline_color=torch.tensor([0.0, 0.0, 0.0], device="cuda"),
        )
        rgb_pd = out['render'].detach().cpu().numpy()
        utils.imwrite(str(output_folder / f"render_{idx:03d}.png"), rgb_pd)

        rgb_visualize = out_vis['render'].detach().cpu().numpy()
        utils.imwrite(str(output_folder / f"render_visualize_{idx:03d}.png"), rgb_visualize)

        rgb_gt = data['rgb'][idx].detach().cpu().numpy()
        utils.imwrite(str(output_folder / f"gt_{idx:03d}.png"), rgb_gt)

        del out, rgb_pd, cam
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Rendered {len(data['camera'])} views to {output_folder}")



if __name__ == "__main__":
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_settings', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--results_folder', type=str, default='./results/unnamed_experiments')
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--just_render', action='store_true')
    parser.add_argument('--render_checkpoint', type=str, default=None)
    parser.add_argument('--render_output_name', type=str, default='all_views')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--experiment_profile_checkpoint', type=str, default=None)
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='PATH=VALUE',
        help=(
            "Override experiment settings after loading JSON files. Repeatable. "
            "Examples: --set total_steps=15000 "
            "--set physics.grid_shape=[64,64,100] "
            "--set physics.rotation_loss_scalings[0].weight=2.0"
        ),
    )
    args = parser.parse_args()

    if args.just_render:
        assert args.render_checkpoint is not None, "Must provide --render_checkpoint when using --just_render"
        assert args.data_folder is not None, "Must provide --data_folder when using --just_render"
        do_renders(
            checkpoint=args.render_checkpoint,
            results_folder=args.results_folder,
            data_folder=args.data_folder,
            device=device,
            output_dir_name=args.render_output_name,
        )
        sys.exit(0)

    if args.experiment_settings is not None:
        with open(args.experiment_settings, 'r') as f:
            experiment_settings_custom = json.load(f)
    else:
        experiment_settings_custom = None

    with open(RELEASE_ROOT / 'experiment_profiles' / 'default_settings.json', 'r') as f:
        experiment_settings = json.load(f)

    if args.experiment_profile_checkpoint is not None:
        checkpoint_settings = load_experiment_settings_from_checkpoint(args.experiment_profile_checkpoint)
        merge_dicts(experiment_settings, checkpoint_settings)

    if experiment_settings_custom is not None:
        merge_dicts(experiment_settings, experiment_settings_custom)

    normalize_experiment_settings(experiment_settings)

    applied_overrides = []
    for override in args.set:
        normalized_path, path_tokens, value = parse_override_assignment(override)
        apply_override(experiment_settings, path_tokens, value)
        applied_overrides.append({
            'path': normalized_path,
            'value': value,
        })

    normalize_experiment_settings(experiment_settings)

    if applied_overrides:
        print("Applied CLI overrides:")
        for override in applied_overrides:
            print(f"  {override['path']} = {json.dumps(override['value'])}")

    if args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = 'teapot7_physics_sweep'

    assert (experiment_settings['physics']['optimize_loss'] == False or experiment_settings['physics']['run_simulation'] == True), "If optimizing physics loss, must also run it"

    if args.data_folder is None:
        raise ValueError('--data_folder must be provided unless using --just_render')

    do_experiment(
        data_folder=args.data_folder,
        experiment_name=experiment_name,
        results_folder=args.results_folder,
        notes=json.dumps(experiment_settings, indent=2),
        experiment_settings=experiment_settings,
        checkpoint=args.checkpoint,
    )

    sys.exit(0)
