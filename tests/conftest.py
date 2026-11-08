import pytest

from data import Catalog
from tests.fixtures import generate


@pytest.fixture(scope="session")
def fixture_directory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("statistical-inputs")
    generate(directory)
    return directory


@pytest.fixture
def catalog(fixture_directory):
    return Catalog(fixture_directory, allow_synthetic=True)
