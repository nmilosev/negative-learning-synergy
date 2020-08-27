# negative-learning-synergy

This repository contains the accompanying code for paper "Synergy between Traditional Classification and Classification Based on Missing Features in Deep Convolutional Neural Networks" submitted to Neural Computing and Applications journal in June 2020.

# Dependencies

- python 3.8
- pytorch 1.6.0 (CUDA supported)
- torchvision 0.7.0 (for CIFAR-10 dataset)

To test the examples we recommend using `pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime` Docker image which can run everything out of the box. ResNet18 model was trained on 1 x NVIDIA Tesla V100 instance. Other models were trained on 1 x NVIDIA RTX2070 instance.

# Running

```
python3 cifar.py

# - or -

python3 cifar-resnet.py
```

# Example outputs

Example outputs are in `.txt` files in this repository.

# Contact

```
nmilosev [at] dmi.rs
```
