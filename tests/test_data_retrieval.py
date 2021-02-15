import datetime
import pytest
import requests
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


@pytest.mark.get_data
def test_get_usbls_data(expected_result_get_usbls_data):
    start_date = datetime.date(year=1995, month=1, day=1)
    end_date = datetime.date(year=1995, month=1, day=1)
    series_ids = ["CUUR0000SA0", "SUUR0000SA0"]

    test_result = data_retrieval.get_usbls_data(
        api_key="mock_key", series_ids=series_ids, start_date=start_date, end_date=end_date
    )
    assert test_result == expected_result_get_usbls_data


@pytest.mark.get_data
def test_get_usbls_data_exception_raised():
    with pytest.raises(requests.HTTPError):
        data_retrieval.get_usbls_data(
            api_key="mock_key",
            series_ids=None,
            start_date=datetime.date(year=2018, month=1, day=1),
        )
