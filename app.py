#!/usr/bin/env python
"""pl-mnist_cnn: train, evaluate, or predict with a small CNN on MNIST digits, as a ChRIS plugin."""

import csv
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from chris_plugin import PathMapper, chris_plugin

from mnist_cnn.config import TrainConfig
from mnist_cnn.data import mnist_loaders
from mnist_cnn.engine import evaluate_model, train_model
from mnist_cnn.models import SimpleCNN
from mnist_cnn.utils import get_device, load_checkpoint, predict_image, resolve_path

__version__ = '1.0.0'

DISPLAY_TITLE = r"""
pl-mnist_cnn: a small CNN for MNIST digits
"""


parser = ArgumentParser(
    description=(
        "Train, evaluate, or predict with a small CNN on MNIST handwritten digits. "
        "In predict mode the input directory holds the images and the checkpoint; "
        "the output directory receives checkpoints, metrics and predictions."
    ),
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True,
                    help="what to do: train a model, evaluate one on the MNIST test set, "
                         "or predict the digit in each input image")

# training
parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="batch size")
parser.add_argument("--lr", type=float, default=TrainConfig.lr, help="learning rate")
parser.add_argument("--seed", type=int, default=TrainConfig.seed, help="random seed")
parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers,
                    help="data loader worker processes; 0 loads in the main process")
parser.add_argument("--device", type=str, default=TrainConfig.device,
                    choices=["auto", "cpu", "cuda", "mps"], help="where to compute")
parser.add_argument("--amp", action="store_true", default=TrainConfig.amp,
                    help="train with mixed precision; has an effect on a CUDA device only")
parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay, help="weight decay")

# eval and predict
parser.add_argument("--weights", type=str, default="best.ckpt",
                    help="checkpoint to load; a relative path is under the input directory")
parser.add_argument("--image", type=str, default="",
                    help="predict this one image; a relative path is under the input directory. "
                         "Without it, every image matching --pattern is predicted")
parser.add_argument("--pattern", type=str, default="**/*.png",
                    help="glob, under the input directory, of the images to predict "
                         "when --image is not given")
parser.add_argument("--suffix", type=str, default=".pred.txt",
                    help="suffix of the per-image result files written in batch prediction")

parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')


def weights_load(options: Namespace, inputdir: Path, device):
    """Build the network and load ``--weights`` into it."""
    weights = resolve_path(options.weights, inputdir)
    if weights is None or not weights.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {weights}. Give --weights; a relative path is under the input directory."
        )
    model = SimpleCNN().to(device)
    load_checkpoint(model, str(weights), map_location=device)
    return model


def train(options: Namespace, outputdir: Path) -> None:
    """Train on MNIST; write ``best.ckpt`` and ``train_metrics.txt`` to the output directory."""
    cfg = TrainConfig(
        epochs=options.epochs,
        batch_size=options.batch_size,
        lr=options.lr,
        seed=options.seed,
        num_workers=options.num_workers,
        out_dir=str(outputdir),
        device=options.device,
        amp=options.amp,
        weight_decay=options.weight_decay,
    )
    loaders = mnist_loaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    history, best_path = train_model(cfg, loaders, SimpleCNN)
    (outputdir / "train_metrics.txt").write_text(
        f"best_ckpt={best_path.name}\n"
        f"last_train_loss={history['train_loss'][-1]:.4f}\n"
        f"last_val_loss={history['validation_loss'][-1]:.4f}\n"
        f"last_val_acc={history['validation_accuracy'][-1]:.2f}\n"
        f"learning_rate={cfg.lr:.5f}\nbatch_size={cfg.batch_size}\nepochs={cfg.epochs}\n"
    )
    print(f"[train] wrote {best_path.name} and train_metrics.txt")


def evaluate(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    """Score a checkpoint on the MNIST test set; write ``eval.txt``."""
    device = get_device(options.device)
    model = weights_load(options, inputdir, device)
    _, test_loader = mnist_loaders(batch_size=options.batch_size, num_workers=options.num_workers)
    loss, acc = evaluate_model(model, test_loader, device)
    (outputdir / "eval.txt").write_text(f"loss={loss:.4f}\nacc={acc:.2f}\n")
    print("[eval] wrote eval.txt")


def predict(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    """Predict the digit in one image, or in every image matching ``--pattern``."""
    device = get_device(options.device)
    model = weights_load(options, inputdir, device)

    if options.image:
        image_path = resolve_path(options.image, inputdir)
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(
                f"No image at {image_path}. A relative --image is under the input directory."
            )
        digit, confidence = predict_image(model, image_path, device)
        (outputdir / "prediction.txt").write_text(
            f"file={image_path.name}\ndigit={digit}\nconfidence={confidence:.2f}\n"
        )
        print("[predict] wrote prediction.txt")
        return

    # Mirror the input tree: in/d1/a.png -> out/d1/a.pred.txt
    mapper = PathMapper.file_mapper(
        inputdir, outputdir, glob=options.pattern, suffix=options.suffix, fail_if_empty=False
    )
    rows = []
    for input_file, output_file in mapper:
        if not input_file.is_file():
            continue
        digit, confidence = predict_image(model, input_file, device)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(f"digit={digit}\nconfidence={confidence:.2f}\n")
        rows.append((str(input_file.relative_to(inputdir)), digit, f"{confidence:.2f}"))

    if not rows:
        raise FileNotFoundError(
            f"No image matched --pattern '{options.pattern}' in the input directory."
        )
    with (outputdir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "predicted_digit", "confidence_percent"])
        writer.writerows(sorted(rows))
    print(f"[predict] {len(rows)} images; wrote results.csv and one '{options.suffix}' file per image")


# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='MNIST digit classifier (small CNN): train, evaluate, predict',
    category='Machine Learning',
    min_memory_limit='2Gi',      # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    Run the mode asked for.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """
    print(DISPLAY_TITLE, file=sys.stderr)
    if options.mode == "train":
        train(options, outputdir)
    elif options.mode == "eval":
        evaluate(options, inputdir, outputdir)
    else:
        predict(options, inputdir, outputdir)


if __name__ == '__main__':
    main()
