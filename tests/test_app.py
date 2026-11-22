"""Tests that need neither the MNIST download nor a trained model."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from app import main, parser
from mnist_cnn.engine import Trainer, evaluate_model
from mnist_cnn.models import SimpleCNN
from mnist_cnn.utils import load_checkpoint, save_checkpoint


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """An input directory holding an untrained checkpoint and three images, and an output directory."""
    inputdir, outputdir = tmp_path / 'incoming', tmp_path / 'outgoing'
    (inputdir / 'scans').mkdir(parents=True)
    outputdir.mkdir()
    torch.manual_seed(0)
    save_checkpoint(SimpleCNN(), inputdir / 'best.ckpt')
    for name, shade in (('a.png', 0), ('b.png', 128), ('scans/c.png', 255)):
        Image.new('L', (40, 30), shade).save(inputdir / name)
    return inputdir, outputdir


def test_predict_batch(dirs: tuple[Path, Path]):
    inputdir, outputdir = dirs
    main(parser.parse_args(['--mode', 'predict', '--device', 'cpu']), inputdir, outputdir)

    rows = (outputdir / 'results.csv').read_text().splitlines()
    assert rows[0] == 'relative_path,predicted_digit,confidence_percent'
    assert [row.split(',')[0] for row in rows[1:]] == ['a.png', 'b.png', 'scans/c.png']
    assert all(0 <= int(row.split(',')[1]) <= 9 for row in rows[1:])
    assert (outputdir / 'scans' / 'c.pred.txt').read_text().startswith('digit=')


def test_predict_single_image(dirs: tuple[Path, Path]):
    inputdir, outputdir = dirs
    options = parser.parse_args(['--mode', 'predict', '--device', 'cpu', '--image', 'scans/c.png'])
    main(options, inputdir, outputdir)

    lines = (outputdir / 'prediction.txt').read_text().splitlines()
    assert lines[0] == 'file=c.png'
    assert lines[1].startswith('digit=') and lines[2].startswith('confidence=')
    assert not (outputdir / 'results.csv').exists()


def test_predict_without_weights_fails(dirs: tuple[Path, Path]):
    inputdir, outputdir = dirs
    options = parser.parse_args(['--mode', 'predict', '--weights', 'missing.ckpt'])
    with pytest.raises(FileNotFoundError, match='No checkpoint'):
        main(options, inputdir, outputdir)


def test_predict_with_no_matching_image_fails(dirs: tuple[Path, Path]):
    inputdir, outputdir = dirs
    options = parser.parse_args(['--mode', 'predict', '--device', 'cpu', '--pattern', '**/*.jpg'])
    with pytest.raises(FileNotFoundError, match='No image matched'):
        main(options, inputdir, outputdir)


def test_training_writes_a_checkpoint_that_loads(tmp_path: Path):
    torch.manual_seed(0)
    images, labels = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    trainer = Trainer(SimpleCNN(), lr=1e-3, weight_decay=0.0, device=torch.device('cpu'),
                      amp=False, out_dir=str(tmp_path))

    history = trainer.train(loader, loader, epochs=2)

    assert len(history['train_loss']) == 2
    reloaded = load_checkpoint(SimpleCNN(), tmp_path / 'best.ckpt')
    loss, acc = evaluate_model(reloaded, loader, torch.device('cpu'))
    assert loss > 0 and 0 <= acc <= 100
