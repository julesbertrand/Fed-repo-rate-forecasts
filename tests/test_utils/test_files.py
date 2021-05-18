import shutil

import pytest

from frp.utils.files import get_valid_filename, open_yaml, save_yaml
from frp.utils.path import create_dir_if_missing


@pytest.mark.parametrize(
    "filename, expected_result",
    [
        ("Date d'envoi", "date_denvoi"),
        (
            "GDP per capita, 1000 of $, Not Seasonally adjusted",
            "gdp_per_capita_1000_of_$_not_seasonally_adjusted",
        ),
        ("It's me", "its_me"),
        ("DATE", "date"),
    ],
)
def test_get_valid_filename(filename, expected_result):
    assert get_valid_filename(filename) == expected_result


@pytest.mark.parametrize("filename", ["", " ", "_", "  _"])
def test_get_valid_filename_errors(filename):
    with pytest.raises(ValueError):
        get_valid_filename(filename)


@pytest.mark.parametrize(
    "data",
    [
        ["provider", "getter", "api_key"],
        list(range(10)),
        {"10": list(range(10)), "6": list(range(6))},
    ],
)
def test_save_and_open_yaml(data):
    dirpath = "./unittests_temp/"
    create_dir_if_missing(dirpath)
    save_yaml(data, dirpath + "test.yaml")
    new_data = open_yaml(dirpath + "test.yaml")
    assert data == new_data
    shutil.rmtree(dirpath)
