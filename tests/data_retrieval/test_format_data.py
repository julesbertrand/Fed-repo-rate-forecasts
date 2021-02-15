import pytest
import pandas as pd
from lib.data_retrieval import format_data


def test_clean_received_data(expected_result_get_fred_data):
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


def test_clean_received_data_exception_raised(expected_result_get_fred_data):
    with pytest.raises(KeyError):
        format_data.clean_received_data(
            expected_result_get_fred_data[0],
            info_data_list=[{}, {}],
            series_cleaner=format_data.clean_fred_series,
        )
