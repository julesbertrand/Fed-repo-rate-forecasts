import pytest
import pandas as pd
from lib import data_retrieval


@pytest.mark.get_data
def test_get_fred_data(expected_result_get_fred_data):
    start_date = "1980-01-08"
    test_params = [
        {"series_id": "FEDFUNDS", "units": "lin", "frequency": "m"},
        {"series_id": "DFF", "units": "lin", "frequency": "m", "aggregation_method": "eop"},
    ]

    test_result = data_retrieval.get_fred_data(
        api_key="mock_key", series_params=test_params, start_date=start_date
    )
    assert test_result == expected_result_get_fred_data


def test_clean_fred_data(expected_result_get_fred_data):
    expected_result = pd.DataFrame(
        {
            "date": {0: pd.Timestamp("1980-01-01 00:00:00")},
            "FEDFUNDS_m_lin": {0: 13.82},
            "DFF_m_lin_eop": {0: 13.82},
        }
    )
    test_result = data_retrieval.clean_fred_data(*expected_result_get_fred_data)

    assert test_result.equals(expected_result)


def test_clean_fred_data_exception_raised(expected_result_get_fred_data):
    with pytest.raises(KeyError):
        data_retrieval.clean_fred_data(expected_result_get_fred_data[0], info_data_list=[{}, {}])


@pytest.fixture
def expected_result_get_fred_data():
    result = (
        [
            {
                "realtime_start": "2021-02-14",
                "realtime_end": "2021-02-14",
                "observation_start": "1980-01-08",
                "observation_end": None,
                "units": "lin",
                "output_type": 1,
                "file_type": "json",
                "order_by": "observation_date",
                "sort_order": "asc",
                "count": 3,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {
                        "realtime_start": "2021-02-14",
                        "realtime_end": "2021-02-14",
                        "date": "1980-01-01",
                        "value": "13.82",
                    },
                ],
            },
            {
                "realtime_start": "2021-02-14",
                "realtime_end": "2021-02-14",
                "observation_start": "1980-01-08",
                "observation_end": None,
                "units": "lin",
                "output_type": 1,
                "file_type": "json",
                "order_by": "observation_date",
                "sort_order": "asc",
                "count": 3,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {
                        "realtime_start": "2021-02-14",
                        "realtime_end": "2021-02-14",
                        "date": "1980-01-01",
                        "value": "13.82",
                    },
                ],
            },
        ],
        [
            {
                "name": "Mock series name",
                "series_id": "FEDFUNDS",
                "frequency": "m",
                "units": "lin",
                "aggregation_method": None,
                "seasonal_adjustment": None,
            },
            {
                "name": "Mock series name",
                "series_id": "DFF",
                "frequency": "m",
                "units": "lin",
                "aggregation_method": "eop",
                "seasonal_adjustment": None,
            },
        ],
    )
    return result
