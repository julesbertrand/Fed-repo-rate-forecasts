import pytest
import pandas as pd
from lib.data_retrieval import format_data


def test_clean_received_data_fred_cleaner(expected_result_get_fred_data):
    expected_result = pd.DataFrame(
        {
            "date": {0: pd.Timestamp("1980-01-01 00:00:00")},
            "FEDFUNDS_m_lin": {0: 13.82},
            "DFF_m_lin_eop": {0: 13.82},
        }
    )
    test_result = format_data.clean_received_data(
        *expected_result_get_fred_data, series_cleaner=format_data.clean_fred_series
    )

    assert test_result.equals(expected_result)

def test_clean_received_data_usbls_cleaner(expected_result_get_usbls_data):
    expected_result = pd.DataFrame(
        {
            "CUUR0000SA0_nsa": {0: 153.5},
            "date": {0: pd.Timestamp("1995-12-01 00:00:00")},
            "SUUR0000SA0_nsa": {0: None},
        }
    ).fillna(.0)
    test_result = format_data.clean_received_data(
        *expected_result_get_usbls_data, series_cleaner=format_data.clean_usbls_series
    ).fillna(.0)
    assert test_result.equals(expected_result)


def test_clean_received_data_exception_raised(expected_result_get_fred_data):
    with pytest.raises(KeyError):
        format_data.clean_received_data(
            expected_result_get_fred_data[0],
            info_data_list=[{}, {}],
            series_cleaner=format_data.clean_fred_series,
        )


@pytest.mark.parametrize(
    "info_data, expected_result",
    [
        (
            {
                "provider": "fred",
                "name": "Effective Federal Funds Rate",
                "series_id": "FEDFUNDS",
                "frequency": "m",
                "units": "lin",
                "aggregation_method": None,
                "seasonal_adjustment": None,
            },
            "FEDFUNDS_m_lin",
        ),
        (
            {
                "provider": "usbls",
                "name": "Labor Force Statistics from the Current Population Survey",
                "series_id": "LNS13000000",
                "frequency": None,
                "units": None,
                "aggregation_method": None,
                "seasonal_adjustment": "sa",
            },
            "LNS13000000_sa",
        ),
        (
            {
                "provider": "usbls",
                "name": "Labor Force Statistics from the Current Population Survey",
                "series_id": "LNS14000000",
                "frequency": None,
                "units": None,
                "aggregation_method": None,
                "seasonal_adjustment": "sa",
            },
            "LNS14000000_sa",
        ),
    ],
)
def test_give_name_to_series(info_data, expected_result):
    test_result = format_data.give_name_to_series(info_data)
    assert test_result == expected_result


@pytest.mark.parametrize("info_data", [{}, {"units": "lin"}])
def test_give_name_to_series_exception_raised(info_data):
    with pytest.raises(KeyError):
        format_data.give_name_to_series(info_data)
