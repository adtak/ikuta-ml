import pytest


@pytest.fixture(scope='session')
def output_dir(tmpdir_factory):
    dir_path = tmpdir_factory.mktemp('test_output')
    return str(dir_path)
