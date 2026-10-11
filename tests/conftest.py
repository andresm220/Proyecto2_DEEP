import pytest
from src.data import TinyHero


@pytest.fixture(scope="session")
def dataset():
    return TinyHero("data", hflip=False)
