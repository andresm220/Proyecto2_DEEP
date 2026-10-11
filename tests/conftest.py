import pytest
from src.data import TinyHero


@pytest.fixture(scope="session")
def dataset():
    return TinyHero("data", hflip=False)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """Run de prueba (1 época, 1 iteración) en CPU: da un checkpoint real con el formato del contrato."""
    import subprocess
    import sys
    out = tmp_path_factory.mktemp("out")
    subprocess.run([sys.executable, "-m", "src.train", "--config", "configs/base.yaml", "--out-root", str(out),
                    "--epochs", "1", "--max-iters", "1"], check=True, capture_output=True)
    return out / "runs" / "base"
