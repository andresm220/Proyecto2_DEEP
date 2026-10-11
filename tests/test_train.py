import csv
import subprocess
import sys

import torch


def run(*args, check=True):
    return subprocess.run([sys.executable, "-m", "src.train", *args], capture_output=True, text=True, check=check)


def test_outputs_and_checkpoint_format(tmp_path):
    run("--config", "configs/base.yaml", "--out-root", str(tmp_path), "--epochs", "1", "--max-iters", "2")
    rd = tmp_path / "runs" / "base"
    with open(rd / "losses.csv") as f:
        assert next(csv.reader(f)) == ["epoch", "iter", "loss_D", "loss_G", "D_x", "D_G_z"]
    ck = torch.load(rd / "checkpoints" / "last.pt", weights_only=False)
    assert {"G", "D", "optG", "optD", "epoch", "config"} <= ck.keys() and ck["epoch"] == 1
    for f in ["fixed_grid.png", "losses.png", "progress.png", "samples/epoch_000.png", "checkpoints/epoch_001.pt"]:
        assert (rd / f).exists(), f
    assert (tmp_path / "fixed_noise.pt").exists()


def test_resume_is_identical_to_uninterrupted(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run("--config", "configs/base.yaml", "--out-root", str(a), "--epochs", "2", "--max-iters", "2")
    run("--config", "configs/base.yaml", "--out-root", str(b), "--epochs", "1", "--max-iters", "2")
    run("--config", "configs/base.yaml", "--out-root", str(b), "--epochs", "2", "--max-iters", "2", "--resume")
    assert (a / "runs/base/losses.csv").read_text() == (b / "runs/base/losses.csv").read_text()


def test_resume_without_checkpoint_fails(tmp_path):
    # --epochs/--max-iters acotan el caso en que el bug existe (arrancaría a entrenar de cero).
    r = run("--config", "configs/base.yaml", "--out-root", str(tmp_path), "--resume",
            "--epochs", "1", "--max-iters", "1", check=False)
    assert r.returncode != 0 and "last.pt" in r.stderr


def test_resume_with_changed_config_fails(tmp_path):
    run("--config", "configs/base.yaml", "--out-root", str(tmp_path), "--epochs", "1", "--max-iters", "1")
    r = run("--config", "configs/exp_hinge.yaml", "--name", "base", "--out-root", str(tmp_path),
            "--epochs", "2", "--max-iters", "1", "--resume", check=False)
    assert r.returncode != 0 and "loss" in r.stderr


def test_load_checkpoint_keeps_rng_state_on_cpu(tmp_path):
    # En Colab device=cuda: los estados RNG deben seguir en CPU o set_rng_state lanza TypeError.
    # "meta" simula un dispositivo que no es CPU sin necesitar GPU.
    from src.train import load_checkpoint
    from src.utils import set_rng_state
    run("--config", "configs/base.yaml", "--out-root", str(tmp_path), "--epochs", "1", "--max-iters", "1")
    ck = load_checkpoint(tmp_path / "runs/base/checkpoints/last.pt", torch.device("meta"))
    assert ck["rng"]["torch"].device.type == "cpu" and ck["loader_rng"].device.type == "cpu"
    set_rng_state(ck["rng"])
    torch.Generator().set_state(ck["loader_rng"])


def test_rerun_without_resume_refuses_to_overwrite(tmp_path):
    args = ["--config", "configs/base.yaml", "--out-root", str(tmp_path), "--epochs", "1", "--max-iters", "1"]
    run(*args)
    csv_path = tmp_path / "runs/base/losses.csv"
    before = csv_path.read_text()
    r = run(*args, check=False)
    assert r.returncode != 0 and "--overwrite" in r.stderr
    assert csv_path.read_text() == before
    run(*args, "--overwrite")
