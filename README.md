# Release

This repository contains the code for our Physics-Guided 3D Gaussian Splatting optimization experiments with Eulerian fluid-based physics losses. It includes the code, experiment profiles, and datasets needed to run experiments for increasing the lift of a plane and making a pourable teapot.

Experiment profiles are in `experiment_profiles/`:
- `experiment_profiles/experiment_profile_plane.json`
- `experiment_profiles/experiment_profile_teapot.json`
- `experiment_profiles/default_settings.json`

The plane profile is intended for the 4 plane types in `data/`, i.e. airbus, b2, paper_plane, and cesna.

The teapot profile can be used with any of `data/teapot_00_ready` through `data/teapot_14_ready`. The example below uses `data/teapot_07_ready`.

## Dependencies

Create a Python environment with Python 3.9+ (tested on Python 3.11.14 with cuda 12.4) and install the release dependencies:

```bash
pip install -r requirements.txt -r requirements-cu124.txt
pip install ./code/torch-splatting/submodules/simple-knn
```

`simple_knn` is a compiled extension and needs its own `pip install` step.


Optional visualization dependencies such as `pyvista`, `plotly`, and `usd-core` are not required for training. If they are missing, the related optional outputs are skipped.

## Example Runs

Plane:

```bash
cd /path/to/repository
python train.py \
  --experiment_settings experiment_profiles/experiment_profile_plane.json \
  --data_folder data/paper_plane_ready \
  --results_folder results \
  --experiment_name plane_release
```

Teapot:

```bash
cd /path/to/repository
python train.py \
  --experiment_settings experiment_profiles/experiment_profile_teapot.json \
  --data_folder data/teapot_07_ready \
  --results_folder results \
  --experiment_name teapot_release
```

## Credits

This release builds on `torch-splatting` by `hbb1` and uses the bundled `simple_knn` CUDA extension included with that codebase. Upstream project: `https://github.com/hbb1/torch-splatting`.

The copied third-party licenses are included in `LICENSE`, `NOTICE`, `code/torch-splatting/LICENSE`, and `code/torch-splatting/submodules/simple-knn/LICENSE.md`.
