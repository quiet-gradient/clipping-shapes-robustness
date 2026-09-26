# Experiment code

Two independent Python packages contain the CIFAR-10/ResNet-18 and
GTSRB/ConvNeXt-T protocols. Use Python 3.11, Linux, CUDA and separate
environments. Run commands from the indicated directory.

The GTSRB evaluation weights are distributed as separate assets of the
`weights-v1` GitHub release, not as Git objects. The 52-file inventory and
SHA-256 digests are in `weights/manifest.json`. To download and verify them:

```bash
gh release download weights-v1 --pattern '*.pt' --dir weights/files
python3 scripts/verify_weights.py --directory weights/files
```

These are model weights for evaluation, not resumable training checkpoints;
they exclude optimizer and RNG state. See `weights/README.md` for scope.
For a GTSRB load/forward check, from `gtsrb_convnextt_cifar_controls/` run
`PYTHONPATH=.:src python scripts/load_release_weight.py ../weights/files/gtsrb_selected_public_pretrain_epoch50_weights.pt`.

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

On Linux, `sha256sum -c MANIFEST.sha256` verifies the shipped files.
