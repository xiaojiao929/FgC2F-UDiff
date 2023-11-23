# FgC2F-UDiff

Official PyTorch implementation of  FgC2F-UDiff described in the paper of "FgC2F-UDiff: Frequency-guided and Coarse-to-fine Unified Diffusion Model for Multi-modality Missing MRI Synthesis"
<img src="./figures/FgC2F-UDiff.png" width="600px">

This repository implements a unified model for cross-modality missing MRI synthesis using a Frequency-guided and Coarse-to-fine Unified Diffusion Model (FgC2F-UDiff) from multiple inputs and outputs. Extensive experimental evaluations across two medical image synthesis datasets demonstrate the effectiveness of FgC2F-UDiff. It consistently generates high-fidelity synthetic images characterized by reduced noise levels, as validated through a comprehensive assessment encompassing both qualitative observations and quantitative metrics. The study provides a new perspective to handle the missing modality issue of current technologies.

## Dependencies

```
python>=3.6.9
torch>=1.7.1
torchvision>=0.8.2
cuda=>11.2
ninja
python3.x-dev (apt install, x should match your python3 version, ex: 3.8)
```

## 1. Python

### 1.1. Installation
Clone `FgC2F-UDiff`:
```shell
git clone https://github.com/xiaojiao929/FgC2F-UDiff
```
Then `cd` into the `FgC2F-UDiff` folder and install it by:
```shell
pip install .
```
**OBS**: The algorithm runs much faster if the compiled backend is used:
```shell
NI_COMPILED_BACKEND="C" pip install --no-build-isolation .
```
However, for running on the GPU, this only works if you ensure that the PyTorch installation uses the same CUDA version that is on your system; therefore, it might be worth installing PyTorch beforehand, *i.e.*:
```shell
pip install torch==1.9.0+cu111
NI_COMPILED_BACKEND="C" pip install --no-build-isolation .
```
where the PyTorch CUDA version matches the output of `nvcc --version`.

