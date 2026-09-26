# Third-Party Software, Data, and Licenses

Original code and author-created assets in this archive are governed by the top-level `LICENSE`. Third-party components remain governed by their respective licenses.

## Redistributed component

| Component | Version | Use | License | Source |
|---|---:|---|---|---|
| AutoAttack | 0.1 | Standard adversarial evaluation | MIT | https://github.com/fra31/auto-attack/tree/a39220048b3c9f2cca9a4d3a54604793c68eca7e |

The bundled wheel packages upstream commit `a39220048b3c9f2cca9a4d3a54604793c68eca7e` and contains its upstream MIT license. Its SHA-256 is `b9b433bcd108a084618ba909e016edde2626a7d8bdda9857a9b7532d4d276cac`. Copyright remains with the upstream author. The wheel's package metadata contains a stale Apache license classifier; the embedded MIT license text is the governing license for the redistributed source.

Please cite Francesco Croce and Matthias Hein, “Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks,” ICML, 2020.

## Runtime dependencies

The binary distributions for these packages are downloaded during environment installation and are not copied into the submission archive. A small Torchvision-derived source adaptation is identified separately below.

| Component | Version | Primary license | License source |
|---|---:|---|---|
| Python | 3.11.x | Python Software Foundation License | https://docs.python.org/3.11/license.html |
| PyTorch | 2.4.1+cu121 | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/v2.4.1/LICENSE |
| Torchvision | 0.19.1+cu121 | BSD-3-Clause | https://github.com/pytorch/vision/blob/v0.19.1/LICENSE |
| NumPy | 1.26.4 | BSD-3-Clause | https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt |
| SciPy | 1.14.1 | BSD-3-Clause with bundled notices | https://github.com/scipy/scipy/blob/v1.14.1/LICENSE.txt |
| Opacus | 1.5.4 | Apache-2.0 | https://github.com/meta-pytorch/opacus/blob/v1.5.4/LICENSE |
| OmegaConf | 2.3.1 | BSD-3-Clause | https://github.com/omry/omegaconf/blob/v2.3.1/LICENSE |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml/blob/6.0.3/LICENSE |
| tqdm | 4.67.1 | MPL-2.0 and MIT | https://github.com/tqdm/tqdm/blob/v4.67.1/LICENCE |

Users should consult the license files installed with each distribution. Transitive binary dependencies may carry additional notices. The NVIDIA driver and CUDA runtime are not distributed in this archive and remain subject to NVIDIA's terms.

## Build dependencies

`pyproject.toml` requests `setuptools>=68` and `wheel` to build the project wheel. Their exact resolved versions are not fixed by this package; both are distributed under MIT-family licenses. They are downloaded by the build environment and are not included in the submission archive.

## Datasets

| Dataset | Distribution in this archive | Source |
|---|---|---|
| CIFAR-10 | Not included; downloaded by the user | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-100 | Not included; downloaded by the user for public pretraining | https://www.cs.toronto.edu/~kriz/cifar.html |

The official CIFAR page provides the datasets, checksums, creators, and required citation, but does not state a standard open-source license for the data. Users must review the source page and determine whether their intended use complies with applicable terms.

The released preparation scripts additionally verify SHA-256 so that the exact downloaded archives are auditable. Expected values are:

| Dataset | Archive | Size | MD5 | SHA-256 |
|---|---|---:|---|---|
| CIFAR-10 | `cifar-10-python.tar.gz` | 170,498,071 bytes | `c58f30108f718f92721af3b95e74349a` | `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |
| CIFAR-100 | `cifar-100-python.tar.gz` | 169,001,437 bytes | `eb9058c3a382ffc7106e4002c42a8d85` | `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` |

Please cite Alex Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” technical report, 2009.

## Architecture implementation

The package constructs ResNet-18 through Torchvision, adapts its stem for CIFAR-sized images, and includes a small adaptation of Torchvision's `BasicBlock.forward` implementation to replace the in-place residual addition. That adapted portion remains governed by Torchvision's BSD-3-Clause license. The required notice follows.

```text
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Initialization checkpoint

The bundled checkpoint is an author-created learned-weight artifact trained on CIFAR-100. The package includes the complete final construction, selection, and export code, while the dataset remains user-downloaded. The top-level package license applies to the checkpoint file to the extent the authors can grant those rights. Use remains subject to any applicable rights and terms associated with the source dataset and upstream software.

Names of upstream projects and authors are used only for attribution. Their inclusion does not imply endorsement of this submission or its conclusions.
