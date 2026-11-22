# pl-mnist_cnn

[![MIT License](https://img.shields.io/github/license/fnndsc/pl-mnist_cnn)](https://github.com/FNNDSC/pl-mnist_cnn/blob/main/LICENSE)
[![ci](https://github.com/FNNDSC/pl-mnist_cnn/actions/workflows/ci.yml/badge.svg)](https://github.com/FNNDSC/pl-mnist_cnn/actions/workflows/ci.yml)

`pl-mnist_cnn` is a [_ChRIS_](https://chrisproject.org/) _ds_ plugin that trains a small
convolutional network to read handwritten digits, scores it, and uses it to label images.

It is the smallest complete machine-learning plugin there is: one model, three modes, a
checkpoint that passes from one run to the next. The task is the MNIST digits, which a
laptop CPU learns to 98% in about a minute, so the whole cycle of train, evaluate and
predict can be run and understood in one sitting. Use it to learn how a model lives in
_ChRIS_, to test a compute environment, or as the skeleton for a plugin whose model matters.

## What it does

| `--mode` | reads | writes |
|---|---|---|
| `train` | nothing from the input directory; downloads MNIST | `best.ckpt`, `train_metrics.txt` |
| `eval` | a checkpoint; downloads MNIST | `eval.txt`: loss and accuracy on the 10,000 test images |
| `predict` | a checkpoint and one or more images | per image, the digit and the model's confidence; `results.csv` |

The modes chain: the output of a `train` run is the input of an `eval` or `predict` run.
In `predict` mode the input directory holds both the checkpoint and the images.

Any image is accepted in `predict` mode; it is converted to greyscale and resized to 28 by 28.
The model was trained on white digits on a black background and expects the same.

`train` and `eval` download the MNIST dataset (about 12 MB) on each run, so they need
network access. `predict` does not.

## Running it

`pl-mnist_cnn` runs from within _ChRIS_ or from the command line. Its command is `mnist_cnn`,
and like every _ds_ plugin it takes an input directory and an output directory.

```shell
mkdir -p in trained scored predicted

# train for two epochs
docker run --rm -v $PWD/in:/incoming:ro -v $PWD/trained:/outgoing \
    ghcr.io/fnndsc/pl-mnist_cnn mnist_cnn --mode train --epochs 2 /incoming /outgoing

# score the checkpoint that run wrote
docker run --rm -v $PWD/trained:/incoming:ro -v $PWD/scored:/outgoing \
    ghcr.io/fnndsc/pl-mnist_cnn mnist_cnn --mode eval /incoming /outgoing

# label images: put PNGs beside best.ckpt, anywhere under the input directory
cp digit-*.png trained/
docker run --rm -v $PWD/trained:/incoming:ro -v $PWD/predicted:/outgoing \
    ghcr.io/fnndsc/pl-mnist_cnn mnist_cnn --mode predict /incoming /outgoing
```

`predicted/results.csv` then reads:

```
relative_path,predicted_digit,confidence_percent
digit-0.png,7,99.99
digit-1.png,2,99.98
```

and beside it, mirroring the input tree, is one result file per image (`digit-0.pred.txt`). With
`--image <file>` only that image is read, and the answer is written to `prediction.txt`.

## Options

| option | default | |
|---|---|---|
| `--mode` | required | `train`, `eval` or `predict` |
| `--epochs` | 3 | training epochs |
| `--batch-size` | 128 | batch size |
| `--lr` | 0.001 | learning rate |
| `--weight-decay` | 0.0 | weight decay |
| `--seed` | 1337 | random seed |
| `--num-workers` | 2 | data loader processes; 0 loads in the main process |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `--amp` | off | mixed precision; has an effect on a CUDA device only |
| `--weights` | `best.ckpt` | checkpoint to load; a relative path is under the input directory |
| `--image` | | predict this one image instead of every match of `--pattern` |
| `--pattern` | `**/*.png` | glob of the images to predict, under the input directory |
| `--suffix` | `.pred.txt` | suffix of the per-image result files |

The published image carries PyTorch's CPU build, so `--device auto` resolves to the CPU there.

## The model

Two 3x3 convolutions (32 and 64 channels), one 2x2 max pool, dropout, and two dense layers
(128 units, then 10). It is trained with AdamW and cross-entropy loss, and the checkpoint kept
is the epoch with the best accuracy on the test set. The code is in [`mnist_cnn/`](mnist_cnn).

## Development

```shell
python -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[dev]'
.venv/bin/pytest
```

The tests need no network and no trained model. To build the image:

```shell
docker build -t localhost/fnndsc/pl-mnist_cnn .
docker run --rm localhost/fnndsc/pl-mnist_cnn mnist_cnn --help
```

A version tag (`1.2.3` or `v1.2.3`) pushed to this repository builds the image and publishes it
to `docker.io/fnndsc/pl-mnist_cnn` and `ghcr.io/fnndsc/pl-mnist_cnn`. The version is
`__version__` in `app.py`.

## Origin

Written by David Edelist as `pl-chrNIST`, for the Boston University EC528 Fall 2025 project
[Federated Learning in Medical Imaging Using the ChRIS Platform](https://github.com/EC528-Fall-2025/FedMed-ChRIS).
It was that project's first milestone: a model training and predicting inside _ChRIS_, before
training was spread across sites with [pl-superlink](https://github.com/FNNDSC/pl-superlink)
and [pl-supernode](https://github.com/FNNDSC/pl-supernode). This plugin does not itself take
part in federated training.
