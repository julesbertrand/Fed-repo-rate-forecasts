import datetime as dt
import pytest
import pandas as pd

from lib.data_retrieval.data_getters import FREDGetter, USBLSGetter, OECDGetter


def test_fredgetter_init():
    getter = FREDGetter(api_key="mock_api_key")
    assert getter is not None


def test_fredgetter_get_multiple_series(expected_result_get_fred_data):
    getter = FREDGetter(api_key="mock_api_key")
    start_date = dt.date(1980, 1, 8)
    end_date = dt.date(2021, 2, 21)
    test_params = [
        {"series_id": "FEDFUNDS", "units": "lin", "frequency": "m"},
        {"series_id": "DFF", "units": "lin", "frequency": "m", "aggregation_method": "eop"},
    ]
    test_result = getter.get_multiple_series(
        series_params=test_params,
        start_date=start_date,
        end_date=end_date
    )
    assert test_result == expected_result_get_fred_data


@pytest.mark.parametrize("series_params, error", [(None, TypeError), ([], ValueError)])
def test_get_fred_data_exception_raised(series_params, error):
    with pytest.raises(error):
        getter = FREDGetter(api_key="mock_api_key")
        getter.get_multiple_series(
            series_params=series_params,
            start_date=dt.date(2018, 1, 1),
        )


@pytest.mark.skip(reason="Not implemented yet")
def test_fredgetter_clean_fred_series(test_data_clean_fred_series):
    getter = FREDGetter(api_key="mock_api_key")
    obs_list, metadata_list, expected_dict = test_data_clean_fred_series
    test_obs_df = getter.clean_received_data(obs_list, metadata_list)
    expected_result = pd.DataFrame.from_dict(expected_dict)
    expected_result["date"] = pd.to_datetime(expected_result["date"])
    assert test_obs_df.equals(expected_result)


def test_usblsgetter_init():
    getter = USBLSGetter(api_key="mock_api_key")
    assert getter is not None


def test_get_usbls_data(expected_result_get_usbls_data):
    getter = USBLSGetter(api_key="mock_api_key")
    start_date = dt.date(1995, 1, 1)
    end_date = dt.date(1995, 1, 1)
    series_ids = ["CUUR0000SA0", "SUUR0000SA0"]
    test_result = getter.get_multiple_series(
        series_params=series_ids, start_date=start_date, end_date=end_date
    )
    print(test_result)
    assert test_result == expected_result_get_usbls_data


@pytest.mark.parametrize("series_params, error", [(None, TypeError), ([], ValueError)])
def test_get_usbls_data_exception_raised(series_params, error):
    with pytest.raises(error):
        getter = USBLSGetter(api_key="mock_api_key")
        getter.get_multiple_series(
            series_params=series_params,
            start_date=dt.date(2018, 1, 1),
        )


def test_oecdgetter_init():
    getter = OECDGetter(api_key="mock_api_key")
    assert getter is not None
