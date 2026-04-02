# Physics-Guides 3D Gaussian Splatting Release Code

This repository contains the code for our Physics-Guided 3D Gaussian Splatting optimization experiments with Eulerian fluid-based physics losses. It includes the code, experiment profiles, and links to datasets needed to run experiments for increasing the lift of a plane and making a pourable teapot.

Experiment profiles are in `experiment_profiles/`:
- `experiment_profiles/experiment_profile_plane.json`
- `experiment_profiles/experiment_profile_teapot.json`
- `experiment_profiles/default_settings.json`

Data is available at https://doi.org/10.17632/xz7tkg2zhd.1

The plane profile is intended for the 4 plane types in `data/` from the above dataset, i.e. airbus, b2, paper_plane, and cesna.

The teapot profile can be used with any of `data/teapot_00_ready` through `data/teapot_14_ready`. The example below uses `data/teapot_07_ready`.

## Dependencies

Create a Python environment with Python 3.9+ (tested on Python 3.11.14 with cuda 12.4) and install the release dependencies:

```bash
pip install -r requirements.txt -r requirements-cu124.txt
pip install ./code/torch-splatting/submodules/simple-knn
```

`simple_knn` is a compiled extension and needs its own `pip install` step.

## Dataset

The supporting dataset used in this project is available at Mendeley Data:

**PG-3DGS Supporting Dataset: Multiview Teapot and Airplane Images, Depth Maps, Masks, and Camera Parameters**  
DOI: https://doi.org/10.17632/xz7tkg2zhd.1

This dataset contains 15 teapots and 4 airplanes, with RGB images, depth maps, masks, and camera parameters for 20 views per object.

To use the data in the below example runs, download, decompress, and place the resulting `data` folder in the root of this repsitory.

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
