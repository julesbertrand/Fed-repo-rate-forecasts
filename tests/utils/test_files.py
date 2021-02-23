import pytest

from lib.utils.files import get_valid_filename


@pytest.mark.parametrize(
    "filename, expected_result", [
        ("Date d'envoi", "date_denvoi"),
        (
            "GDP per capita, 1000 of $, Not Seasonally adjusted",
            "gdp_per_capita_1000_of__not_seasonally_adjusted"
        ),
        ("It's me", "its_me"),
        ("DATE", "date")
    ]
)
def test_get_valid_filename(filename, expected_result):
    assert get_valid_filename(filename) == expected_result

@pytest.mark.parametrize(
    "filename", ["", " ", "_", "  _",]
)
def test_get_valid_filename_errors(filename):
    with pytest.raises(ValueError):
        get_valid_filename(filename)
