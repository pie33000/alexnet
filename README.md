---
license: mit
# AlexNet ImageNet Training

## 1. Introduction
This repository contains a **from-scratch PyTorch implementation of AlexNet** trained on the ImageNet-1K dataset. It reproduces the classic 2012 network with modern training utilities such as data augmentation, learning-rate warm-up, and cosine/step decay scheduling.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/60/AlexNet.svg" width="550"/>
</p>

## 2. Project Structure
```
├── model.py          # AlexNet architecture (5 conv + 3 fc)
├── load_data.py      # ImageNet dataloaders & preprocessing
├── train.py          # Training / validation loop & scheduler setup
├── models/           # (auto-created) checkpoints & logs
└── README.md         # You are here
```

### `model.py`
* **Features block** – 5 convolutional layers:
  1. 96 × \(11\times11\) conv, stride 4  
  2. 256 × \(5\times5\) conv, padding 2  
  3. 384 × \(3\times3\) conv, padding 1  
  4. 384 × \(3\times3\) conv, padding 1  
  5. 256 × \(3\times3\) conv, padding 1  
* **Classifier** – flatten → 4096 → 4096 → 1000 with ReLU and Dropout.
* Optional Kaiming/Xavier weight initialisation via `--init_weights`.

### `load_data.py`
* **Training augmentations** – resize shorter side to 256 px → random 224-px crop → horizontal flip.
* **Validation augmentations** – resize 256 px → **TenCrop(224)** (5 crops + mirror) → normalisation.
* Returns two PyTorch `DataLoader`s.

### `train.py`
* Implements the epoch/iteration loop, loss backwards pass, accuracy calculation and checkpointing.
* Supports **learning-rate warm-up** for the first *N* epochs (`--warmup_epochs`).
* Choose between **step decay** or **cosine annealing** via `--scheduler`.
* Logs Top-1 accuracy & loss to `models/top1_accuracy.txt` and saves a checkpoint every 10 epochs.

## 3. Dataset
The code expects the ImageNet directory in the original layout:
```
ILSVRC2012
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
```
Pass the root directory with `--root /path/to/ILSVRC2012`.

> 💡 **ImageNet licence** – obtaining the dataset requires registration with the ImageNet website.

## 4. Installation
```bash
# (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# or the CUDA wheels if you have a GPU
```

## 5. Training
Run:
```bash
python train.py \
  --root /datasets/ILSVRC2012 \
  --device cuda:0             # or cpu / mps
```

Common flags:
* `--epochs` (default 100)
* `--batch_size` (default 128)
* `--lr`, `--momentum`, `--weight_decay`
* `--scheduler` `step|cosine` + `--lr_step_size`, `--lr_gamma`
* `--warmup_epochs` – linear warm-up length
* `--save_dir` – directory for checkpoints & logs

### Download a Pre-Trained model

https://huggingface.co/Pie33000/alexnet

### Resuming / fine-tuning
To resume from a checkpoint:
```bash
python train.py --root /datasets/ILSVRC2012 --device cuda \
                --init_weights False \
                --save_dir models \
                --epochs 30
# then inside train.py adapt: model.load_state_dict(torch.load('models/model_XX.pth'))
```

## 6. Metrics
The script prints **Top-1 Accuracy** after every epoch. You can extend it to Top-5 with:
```python
maxk = 5
_, pred = logits.topk(maxk, 1, True, True)  # (batch, 5)
correct = pred.eq(labels.view(-1, 1).expand_as(pred))
correct_top5 += correct.any(1).float().sum().item()
```

## 7. Citation
If you use this code in your research, please cite:
> Krizhevsky, Alex, Ilya Sutskever, and Geoffrey Hinton. "ImageNet classification with deep convolutional neural networks." *NeurIPS* 2012.

## 8. License
license: mit
## Download the data
    mkdir data && cd data
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

## Train model

    conda create -n alexnet python=3.12
    conda activate alexnet
    pip install -r requirements.txt
    python train.py --root /root/alexnet/data --device cuda

## Evaluate model
    python evaluate.py --root /root/alexnet/data --device cuda --model_path models/model_{epoch}.pth
