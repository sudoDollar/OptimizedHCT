# OptimizedHCT
ECE-GY 9143 - High Performance Machine Learning Project (NYU)

## Description
The goal of this project is to optimize the [1]High-resolu1on Convolu1onal Transformer (HCT) model by
leveraging High-Performance Computing (HPC) techniques, including distributed data parallelism and
other advanced op1miza1on strategies. By enhancing the efficiency and scalability of HCT, the project
aims to enable faster and more effective processing of high-resolu1on images in applica1ons such as
medical imaging and satellite image analysis, ultimately advancing the state-of-the-art in image-based
tasks through optimized model architectures and computational methodologies.

## Wandb Link
[Weight & Biases](https://wandb.ai/ar7996/HCT)

## Project Milestones
### Milestone

| Milestone | Status |
| --- | --- |
| Write data module to handle the dataset (load, pre-process, make it pytorch compatible) | Completed |
| Write a basic trainer as per model paper to measure time and accuracy of model on single GPU. | Completed |
| Integrate Weights & Biases and PyTorch Profiler | Completed |
| Optimize training time by using PyTorch distributed data parallel | Completed |
| Make it torchscript compatible to make it deployable on non-python environments | Completed |
| Use additional quantization techniques to reduce inference time further | Stucked due to PyTorch Limitations |


## Repository Structure
```
├── ...
├── cpp
│ ├── native_hct.cpp (To load TorchScript Model using C++)
|
|── layer
| ├── ac_layers.py (Attention-Convolution block for Transformer)
| ├── performer_attention.py (Linear Self-Attention)
| ├── resnet_layers.py (Convolution layers for early stages)
|
|── dataset
| ├── dataset/image_labels.csv (For PyTorch Dataset class)
| ├── dataset/image_labels_valid.csv (For PyTorch Dataset class)
|
|── hct_base.py (Model Starting point)
|── datascript.py (To process data and make PyTorch compatible)
|── data.py (Dataset class for PyTorch Dataloader)
|── trainer.py (Single Device Trainer)
|── DistributedTrainer.py (Distributed Data Parallel Trainer)
|── inference.py (Inference Engine)
|── train.py (Run this file to train the model)
|── inf.py (Use this file for inference purpose)
|── params.py (For command line arguments)
|── submit.sh (To submit job on HPC)
|── submit2.sh (To submit job on HPC)

```

## Steps to Run

### Required Softwares/Libraries
1. Pytorch
2. Wandb
3. PyTorch LibTorch (For C++) (with CUDA Support)
4. OpenCV C++
5. Torch_tb_profiler
6. einops (Python Library for Array Manuplation)

### Training
#### Using HPC
```
sbatch submit.sh
```

#### Normal Mode
```
python train.py

usage: train.py [-h] [--batch-size N] [--num-workers NUM_WORKERS] [--epochs N] [--lr LR] [--device DEVICE] [--optimizer OPTIMIZER] [--momentum N]
                [--weight-decay N] [--dataset-path DATASET_PATH] [--wandb WANDB] [--project PROJECT] [--training-mode TRAINING_MODE] [--num-devices NUM_DEVICES]
                [--log-path LOG_PATH]

High Resolution Convolutional Transformer

options:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 16)
  --num-workers NUM_WORKERS
                        Number of I/O processes (default: 2)
  --epochs N            Number of epochs to train (default: 80)
  --lr LR               learning rate (default: 2e-5)
  --device DEVICE       Device to be used for training(cpu/cuda)
  --optimizer OPTIMIZER
                        Optimizer to be used (default: Adam)
  --momentum N          Momentum Value (default: 0.9)
  --weight-decay N      Weight Decay value (default 5e-4)
  --dataset-path DATASET_PATH
                        Path of LIU4K Dataset (default: ./dataset)
  --wandb WANDB         Use Weight & Biases to track training progress (0/1)
  --project PROJECT     Project Name for W&B
  --training-mode TRAINING_MODE
                        Training Mode (Parallel/Distributed/Single)
  --num-devices NUM_DEVICES
                        Number of Learners/GPUs
  --log-path LOG_PATH   Path where logs will be captured
```

#### Note: Need to change wandb API key in submit.sh file

### Inference

#### Using HPC
```
sbatch submit2.sh
```

#### Normal Mode
```
python inf.py
```


### C++
```
cd cpp
mkdir build
cd build
cmake ..
cmake --build .
./native_hct <model_path> <image_path> <device>
```

## Results

| Per Epoch | Single Device | Distributed (4 GPUs) | SpeedUp |
| --- | --- | --- | --- |
| Data Loading | 1000 secs | 271 secs | 3.7 |
| Training | 160 secs | 90 secs | 1.78 |
| Running | 1170 secs | 360 secs | 3.25 |

#### Note:
**Training => Data Movement from CPU to GPU + Forward pass + Backward Pass + Metrics Calculation** \
**Running => Data Loading + Training** \
**Model Evaluation is not included in time calculation**

![Image]('misc/W&B Chart 5_10_2024, 6_43_40 PM.png' "Title") \
![Image]("misc/W&B Chart 5_10_2024, 6_44_18 PM.png") \
![Image]("misc/W&B Chart 5_10_2024, 6_44_29 PM.png") \
![Image]("misc/W&B Chart 5_10_2024, 6_44_38 PM.png") \
![Image]("misc/W&B Chart 5_10_2024, 6_44_45 PM.png") 



## References
[1] Taha, A., Truong Vu, Y.N., MombourqueQe, B., MaQhews, T.P., Su, J. and Singh, S., 2022, September.
Deep is a Luxury We Don’t Have. MICCAI 2022
