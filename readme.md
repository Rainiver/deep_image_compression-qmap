# Deep Learning Image Compression

This repository provides an experimental framework for **image compression based on deep neural networks**.  
It integrates modules for entropy coding, JPEG tools, training pipelines, and evaluation scripts.

The framework is designed for research purposes and demonstrates how classical codecs and learned compression can be combined.

---

## Features

- 📦 End-to-end deep learning image compression  
- ⚡ Custom entropy coding modules (C/C++ extensions)  
- 🖼️ JPEG decoder utilities for preprocessing and quantization table extraction  
- 🔄 Training & testing pipelines with configurable experiments  
- 📊 Built-in evaluation: PSNR, MS-SSIM, rate–distortion (RD) curves  
- 🧩 Extensible with quantization-aware training, TorchJPEG, and mixed precision (Apex)

---

## Getting Started

> **Note:** This README keeps the original workflow and commands but removes internal-only links. Adjust paths and cluster commands to match your environment.

### 1. Build native dependencies

Compile entropy coding modules:

```bash
# Activate the environment used in the original project (example)
source s0.3.2

# Build C/C++ entropy coding module
cd codes/cc
make
cd ../..
```
Compile JPEG decoder tools (binary + `.so`) and generate quantization tables:
```bash
source s0.3.2
cd jpeg/jpeg_decoder
make
python get_QT.py
cd ../..
```
---
### 2. Python Dependencies

Install required Python packages:

```bash
source s0.3.2
pip install --upgrade pip --user
pip install tensorboard --user
pip install -r requirements.txt --user
```

Optional:

* **TorchJPEG**: for DCT-related experiments
* **NVIDIA Apex**: for mixed precision training

---

## Training

Start training:

```bash
source s0.3.2
export PYTHONPATH=.:$PYTHONPATH

cd tools
bash train.sh spring_scheduler ../experiments/GG18/
```

To run a full RD curve, see the scripts in `tools/auto_train.sh` and `tools/auto_test.sh`.

Quantization-aware training example:

```bash
git submodule update --init
bash train_quant.sh ../experiments/1dn_GG18_quantV1/ ../experiments/integer_configs/warm_w8_a8.yaml
```

---

## Monitoring Training

Launch TensorBoard:

```bash
tensorboard --logdir experiments --host 0.0.0.0 --port 16384
```

View results at `http://<host-ip>:16384/`.

---

## Auto Training & Testing

### Auto Train

`tools/auto_train.py` supports automated training across multiple loss functions and configurations.

Example arguments:

* `-tp` / `--loss_weight_parameter_type`: choose from `psnr`, `msssim`, `hybrid`, `grad`, etc.
* `-pi` / `--input_dir_name`: input model path
* `-re` / `--restore_dir`: restore checkpoint directory

### Auto Test

`tools/auto_test.py` supports:

1. Merging validation results across models
2. Batch testing on datasets
3. Merging and exporting results to CSV

---

## Model Export

### To Caffe

```bash
git submodule update --init
cd nart/python
python setup.py install
cd tools
bash to_caffe.sh VI_AIC_TITANXP ../exp_dir
```

For parameter/FLOPs statistics:

```bash
python -m spring.nart.tools.caffe.count caffe/y_decoder.prototxt
```

### To TensorRT (via NART)

Requires CUDA 10, TensorRT 7, and Python 3.6:

```bash
bash to_nart.sh
```

---

## Testing

Single-image testing:

```python
from tools.test import main

main(
    img_path="example.png",
    base=64,
    log_dir="../experiments/my_model/",
    epoch=50
)
```

Folder testing:

```python
from tools.test import test

test(data_dir="path/to/validation/images")
```

GPU testing:

```bash
./test.sh VI_AIC_1080TI
```

---

## Pipeline & Playground

The entry point for new experiments is:

```bash
python tools/playground.py
```

Supported modes:

* `train`
* `test`
* `compress`
* `decompress` (WIP)
* `to_caffe` (WIP)

Pipeline configurations are YAML-based, supporting modular process definitions and dynamic model builders.
