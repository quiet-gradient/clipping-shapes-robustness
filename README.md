# Experiment code

Two independent Python packages contain the CIFAR-10/ResNet-18 and
GTSRB/ConvNeXt-T protocols. Use Python 3.11, Linux, CUDA and separate
environments. Run commands from the indicated directory.

CIFAR and GTSRB evaluation weights are distributed as assets of the
`weights-v1` GitHub release. The 140-file inventory and SHA-256 digests are in
`weights/manifest.json` and `weights/additional_manifest.json`. To download and verify them:

```bash
gh release download weights-v1 --repo quiet-gradient/clipping-shapes-robustness \
  --pattern '*.pt' --dir weights/files
python3 scripts/verify_weights.py --directory weights/files
```

These are model weights for evaluation, not resumable training checkpoints;
they exclude optimizer and RNG state. See `weights/README.md` for scope.
For a GTSRB load/forward check, from `gtsrb_convnextt_cifar_controls/` run
`PYTHONPATH=.:src python scripts/load_release_weight.py ../weights/files/gtsrb_selected_public_pretrain_epoch50_weights.pt`.
For CIFAR, from `cifar10_resnet18/`, run
`PYTHONPATH=src python scripts/load_release_weight.py ../weights/files/cifar_hce_c030_lr0050_s42_weights.pt`.
Both loaders accept unnormalized [0,1] images and apply normalization once.

## CIFAR-10 / ResNet-18 (`cifar10_resnet18/`)

```bash
cd cifar10_resnet18
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python scripts/prepare_data.py --data-root data
DATA_ROOT="$PWD/data" python scripts/run_experiments.py --all --dry-run
DATA_ROOT="$PWD/data" python scripts/run_experiments.py --all
DATA_ROOT="$PWD/data" python scripts/run_autoattack.py --all
```

`artifacts/public_pretrain/checkpoint_epoch_170.pt` is included. To rerun
public pretraining, use `scripts/prepare_pretraining_data.py` and
`scripts/run_pretraining.py`. The frozen training, attack and test rosters
are in `manifests/` and `configs/`. Official-test evaluation uses
`python scripts/run_official_test.py --all --data-root data --acknowledge-official-test`
only after the checkpoints and choices are fixed.

## GTSRB / ConvNeXt-T and CIFAR controls (`gtsrb_convnextt_cifar_controls/`)

```bash
cd gtsrb_convnextt_cifar_controls
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -r requirements-test.txt
python -m pip install --no-deps -e .
PYTHONPATH=.:src python -m pytest -q tests/test_gtsrb_outer_kl.py tests/test_privacy.py
PYTHONPATH=.:src python -m trustml.ssg_gtsrb.rectangular_v2 --dry-run
```

Scientific training and attack code is under `src/trustml/`; frozen run
parameters and rosters are under `configs/`. GTSRB launchers still require
the original platform's routes/artifact adapter, external datasets and the
selected public pretrain checkpoint. This source tree alone is **not** a
turnkey GPU rerun of the GTSRB campaign. The separate release contains
evaluation weights, but not optimizer/RNG state, test-result masks, private
data, or the original platform routes.
The hard lock's `executed_base_file_sha256` identifies the original training
config; the released, administratively cleaned copy has its own hash in
`MANIFEST.sha256`.

## Additional experiments

Appendix E adds fixed-norm N, U/M at epsilon 2, uniform attenuation cM, a non-private baseline and aggregate-norm trajectories. In `gtsrb_convnextt_cifar_controls/`, code is in `src/trustml/gradient_controls/`, run configurations in `configs/gradient_controls/runs/`, and numerical data in `reports/gradient_controls/`.

From that package directory, with the environment above:

```bash
python -m pip install -r requirements-controls.txt
python scripts/analyze_gradient_controls.py
```

This writes CSV statistics to `outputs/gradient_controls/`. Accuracy differences are in percentage points; SD is computed across training seeds.

For CIFAR, place `cifar-10-python.tar.gz` and extracted `cifar-10-batches-py/` in `data/`. Public initialization is included in `../cifar10_resnet18/artifacts/public_pretrain/`. Example training and test evaluation:

```bash
control_config=configs/gradient_controls/runs/rb_cifar_hard_n_eps4_s45.yaml
PYTHONPATH=.:src python -m trustml.gradient_controls.cifar_runtime \
  --config "$control_config" --output outputs/cifar_n --private-state-dir protected/cifar_n
PYTHONPATH=.:src python -m trustml.gradient_controls.cifar_aa \
  --config "$control_config" --checkpoint outputs/cifar_n/checkpoints/final.pt \
  --output outputs/cifar_n_test --split test
```

Use a distinct output/state directory per run. For each epsilon-2 U/M pair, pass the same `--private-root-file` to both training commands, with separate keys per map/seed. Keep `protected/` local. The non-private seed-42 LR search uses `--split dev`; `nonprivate_locked` uses the selected LR .05 for test evaluation. cM reads the supplied calibration files.

For GTSRB, prepare the public initialization and official archives, then run the six N configurations:

```bash
gh release download weights-v1 --repo quiet-gradient/clipping-shapes-robustness \
  --pattern gtsrb_selected_public_pretrain_epoch50_weights.pt --dir ../weights/files
PYTHONPATH=.:src python scripts/prepare_control_inputs.py \
  --gtsrb-training-archive /path/to/GTSRB_Final_Training_Images.zip \
  --gtsrb-test-images /path/to/GTSRB_Final_Test_Images.zip \
  --gtsrb-test-labels /path/to/GTSRB_Final_Test_GT.zip
control_config=configs/gradient_controls/runs/rb_gtsrb_hard_n_eps4_s45.yaml
PYTHONPATH=.:src python -m trustml.gradient_controls.gtsrb_n_runtime \
  --config "$control_config" --output outputs/gtsrb_n --private-state-dir protected/gtsrb_n
PYTHONPATH=.:src python -m trustml.gradient_controls.gtsrb_n_runtime \
  --config "$control_config" --evaluate-checkpoint outputs/gtsrb_n/checkpoints/final.pt \
  --output outputs/gtsrb_n_test
```

On Linux, `sha256sum -c MANIFEST.sha256` verifies the shipped files.
