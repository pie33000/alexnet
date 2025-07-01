# alexnet
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