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

# 2. Step-by-Step Guide 
## 2.1 Prerequisites
Before starting, ensure you have the following:

--A machine with a compatible GPU (NVIDIA recommended) and sufficient VRAM (>=8GB recommended).

--Python 3.6.9 or higher installed.

--An environment set up with conda or venv (recommended for dependency isolation).

## 2.2 Clone the Repository
```
git clone https://github.com/xiaojiao929/FgC2F-UDiff.git
cd FgC2F-UDiff
```

## 2.3 Set Up the Environment
--Install the required dependencies:
```
pip install -r requirements.txt
```

(This installs all necessary libraries such as PyTorch, torchvision, and other related dependencies.
If requirements.txt includes specific CUDA or PyTorch versions, verify compatibility with your GPU and adjust the installation if needed.)

## 2.4 Prepare the Dataset
-Download the Dataset: Follow the dataset download instructions (if available in the repository documentation or README).

-Organize the Dataset: Ensure the dataset is structured correctly. Use the recommended directory structure:
```
./data/
    ├── train/
    ├── test/
    ├── validation/
```

-Preprocess the Data:
```
python preprocess.py --input_dir ./data --output_dir ./processed_data
```

## 2.5 Train the Model
-To train the model from scratch:
```
python train.py --config configs/default.yaml
```
(The --config option specifies the configuration file. Modify the default.yaml file as needed to adjust hyperparameters such as learning rate, batch size, and epochs.)

-If resuming training from a checkpoint:
```
python train.py --config configs/default.yaml --resume ./checkpoints/checkpoint.pth
```

## 2.6 Test/Evaluate the Model
- To evaluate the trained model:
```
  python evaluate.py --config configs/default.yaml --checkpoint ./checkpoints/best_model.pth
```
(Replace best_model.pth with the actual checkpoint file name if different.)

-Specify the test dataset path if not pre-configured in the YAML file:
```
python evaluate.py --config configs/default.yaml --test_data ./processed_data/test
```

## 2.7 Generate Results
-To use the trained model for inference (e.g., synthesizing medical images):
```
python inference.py --config configs/default.yaml --checkpoint ./checkpoints/best_model.pth --input ./sample_input --output ./generated_results
```
(--Replace ./sample_input with the directory containing input images.
--The generated results will be saved in the ./generated_results folder.)

## 2.8 Visualize the Results
-Use the visualization scripts provided:
```
python visualize.py --results_dir ./generated_results --save_dir ./visualized_output
```

-Alternatively, you can use tools like Matplotlib to inspect and analyze the results interactively:
```
python -m scripts.plot_results --results ./generated_results
```

## 2.9 Reproducing the Reported Metrics
-Run the evaluation script to compute PSNR, SSIM, or other metrics:
```
python metrics.py --ground_truth ./processed_data/test --predictions ./generated_results
```

-Ensure the paths for ground_truth and predictions point to the correct directories.

## 2.10 Troubleshooting
-Check the README.md or docs folder for additional information on specific commands or arguments.

-Verify your environment setup using:
```
python check_env.py
```

-For any errors or issues, consult the repository's GitHub Issues page or open a new issue.

## Notes
-The YAML configuration files (e.g., configs/default.yaml) are central to defining the training and inference parameters. Modify them as needed for your experiments.
-If any pretrained models are available, download them using the provided links and place them in the checkpoints directory.

