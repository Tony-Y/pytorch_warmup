# EMNIST Example

Requirements: `pytorch_warmup` and `torchvision`.

<p align="center">
  <img src="https://github.com/Tony-Y/pytorch_warmup/raw/master/examples/emnist/figs/accuracy.png" alt="Accuracy" width="400"/></br>
  <i>Test accuracy over time for each warmup schedule.</i>
</p>

<p align="center">
  <img src="https://github.com/Tony-Y/pytorch_warmup/raw/master/examples/emnist/figs/learning_rate.png" alt="Accuracy" width="400"/></br>
  <i>Learning rate over time for each warmup schedule.</i>
</p>

## Download EMNIST Dataset

Run the Python script `download.py` to download the EMNIST dataset:

```shell
python download.py
```

This script shows download progress:

```
Downloading zip archive
Downloading https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip to .data/EMNIST/raw/gzip.zip
100.0%
```

## Train A CNN Model

Run the Python script `main.py` to train a CNN model on the EMNIST dataset using the Adam algorithm.

### Untuned Linear Warmup

Train a CNN model with the *Untuned Linear Warmup* schedule:

```
python main.py --warmup linear
```

### Untuned Exponential Warmup

Train a CNN model with the *Untuned Exponential Warmup* schedule:

```
python main.py --warmup exponential
```

### RAdam Warmup

Train a CNN model with the *RAdam Warmup* schedule:

```
python main.py --warmup radam
```

### No Warmup

Train a CNN model without warmup:

```
python main.py --warmup none
```

### Usage

```
usage: main.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR]
               [--lr-min LM] [--wd WD] [--beta2 B2] [--no-cuda] [--seed S]
               [--log-interval N] [--warmup {linear,exponential,radam,none}] [--save-model]

PyTorch EMNIST Example

options:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               base learning rate (default: 0.01)
  --lr-min LM           minimum learning rate (default: 1e-5)
  --wd WD               weight decay (default: 0.01)
  --beta2 B2            Adam's beta2 parameter (default: 0.999)
  --no-cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training status
  --warmup {linear,exponential,radam,none}
                        warmup schedule
  --save-model          For Saving the current Model
```

&copy; 2024 Takenori Yamamoto