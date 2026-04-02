# Physics-Guided 3D Gaussian Splatting Official Implementation

**Authors:** Zachary Lee, Maxwell Jacobson, Yexiang Xue

This repository contains the implementation for our Physics-Guided 3D Gaussian Splatting optimization experiments with Eulerian fluid-based physics losses. It includes the code, experiment profiles, and links to datasets needed to run experiments for increasing the lift of a plane and making a pourable teapot.

Experiment profiles are in `experiment_profiles/`:
- `experiment_profiles/experiment_profile_plane.json`
- `experiment_profiles/experiment_profile_teapot.json`
- `experiment_profiles/default_settings.json`

The plane profile is intended for the 4 plane types in `data/` from the above dataset, i.e. airbus, b2, paper_plane, and cessna.

The teapot profile can be used with any of `data/teapot_00_ready` through `data/teapot_14_ready`. The example below uses `data/teapot_07_ready`.

## Dependencies

Create a Python environment with Python 3.9+ (tested on Python 3.11.14 with CUDA 12.4) and install the release dependencies, for example with `conda`:

```bash
conda create -n pg3dgs python=3.11 -y
conda activate pg3dgs
conda install cuda cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti -c nvidia/label/cuda-12.4.0 -y # omit this if you have cuda 12.4 installed already
conda install gcc=13.2 gxx=13.2 -y
pip install -r requirements.txt -r requirements-cu124.txt # or install torch according to your specific cuda version (untested)
pip install ./code/torch-splatting/submodules/simple-knn # try with --no-build-isolation if you get an error about torch not being installed
pip install 'numpy<2' # include this if matplotlib gives an error about numpy version
```

This repository builds a custom CUDA extension (`simple_knn`). For the supported configuration, install a PyTorch build for CUDA 12.4 and use a local CUDA 12.4 toolkit when compiling the extension. Other CUDA 12.x combinations may work if the PyTorch build and local toolkit are aligned, but they are not officially tested in this release.

## Dataset

The supporting dataset used in this project is available at Mendeley Data:

**PG-3DGS Supporting Dataset: Multiview Teapot and Airplane Images, Depth Maps, Masks, and Camera Parameters**  
DOI: https://doi.org/10.17632/xz7tkg2zhd.1

This dataset contains 15 teapots and 4 airplanes, with RGB images, depth maps, masks, and camera parameters for 20 views per object.

To use the data in the example runs below, download and extract the dataset, and place the resulting `data` folder in the root of this repository.

## Example Runs

It is recommended to have at least 24GB of VRAM to run these experiments (Tested on an NVIDIA A30).

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
