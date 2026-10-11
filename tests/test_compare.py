import subprocess
import sys


def test_compare_makes_figures(tmp_path):
    for name in ["r1", "r2"]:
        subprocess.run([sys.executable, "-m", "src.train", "--config", "configs/base.yaml", "--name", name,
                        "--out-root", str(tmp_path), "--epochs", "1", "--max-iters", "2"], check=True)
    out = tmp_path / "cmp"
    subprocess.run([sys.executable, "-m", "src.compare", "--out-root", str(tmp_path),
                    "--runs", "r1", "r2", "--out", str(out)], check=True)
    assert (tmp_path / "cmp_grids.png").exists() and (tmp_path / "cmp_curves.png").exists()
