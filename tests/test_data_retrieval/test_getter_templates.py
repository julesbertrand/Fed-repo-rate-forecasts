import datetime as dt

import pytest
import requests

from lib.data_retrieval.getter_templates import MinimalGetter, TemplateGetter


@pytest.mark.parametrize(
    "getter, kwargs",
    [
        (MinimalGetter, {}),
        (
            TemplateGetter,
            {
                "api_key": "mock_api_key",
                "api_endpoint": "mock_api_endpoint",
                "date_format": "mock_date_format",
            },
        ),
    ],
)
def test_getter_init(getter, kwargs):
    """Test instaciation of Getter"""
    obj = getter(**kwargs)
    assert obj is not None


def test_minimalgetter_check_response():
    """Test check response, reused in all getters classes"""
    getter = MinimalGetter()
    # pylint: disable=protected-access
    response = requests.get(url="test_url")
    with pytest.raises(requests.HTTPError):
        getter._check_response_status_code(response)


@pytest.mark.parametrize(
    "date, date_format, parsed_date",
    [
        (dt.date(1966, 1, 23), "%Y-%m-%d", "1966-01-23"),
        (dt.date(1966, 1, 23), "%Y-%m", "1966-01"),
        (dt.date(9999, 12, 31), "%Y-%m-%d", "9999-12-31"),
        (dt.date.today(), "%Y-%m-%d", dt.date.today().strftime("%Y-%m-%d")),
    ],
)
def test_parse_date(date, date_format, parsed_date):
    """Test parse date function"""
    # pylint: disable=protected-access
    getter = TemplateGetter(
        api_key="mock_api_key", api_endpoint="mock_api_endpoint", date_format=date_format
    )
    assert getter._parse_date(date) == parsed_date


def test_give_name_to_series(test_data_give_name_to_series):
    """Test name attribution"""
    getter = TemplateGetter(
        api_key="mock_api_key", api_endpoint="mock_api_endpoint", date_format="mock_date_format"
    )
    series_info, expected_result = test_data_give_name_to_series
    # pylint: disable=protected-access
    test_result = getter._give_name_to_series(series_info)
    assert test_result == expected_result


@pytest.mark.parametrize("info_data", [{}, {"units": "lin"}])
def test_give_name_to_series_exception_raised(info_data):
    """Test name attribution for wrong inputs: raise error"""
    getter = TemplateGetter(
        api_key="mock_api_key", api_endpoint="mock_api_endpoint", date_format="mock_date_format"
    )
    with pytest.raises(KeyError):
        # pylint: disable=protected-access
        getter._give_name_to_series(info_data)
