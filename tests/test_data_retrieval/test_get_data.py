import shutil

import pytest

from frp.config import API_REQUESTS_PARAMS_FILEPATH
from frp.data_retrieval.get_data import get_data_from_apis
from frp.utils.errors import InvalidAPIKey, InvalidAPIRequestsParams
from frp.utils.files import open_file

API_MOCK_KEYS = {"FRED": "mock_key", "USBLS": "mock_key", "OECD": "mock_key"}


def test_get_data_fred():
    """Test the whole get data workflow for the FREDGetter"""
    api_keys = API_MOCK_KEYS
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    api_params["FRED"]["series_params"] = api_params.get("FRED").get("series_params")[:2]
    _ = get_data_from_apis(
        api_keys=api_keys,
        api_params=api_params,
        data_start_date=data_start_date,
        providers=["FRED"],
        save_dirpath="./unittests_temp/",
    )
    shutil.rmtree("./unittests_temp/")


def test_get_data_usbls():
    """Test the whole get data workflow for the USBLSGetter"""
    api_keys = API_MOCK_KEYS
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    api_params["USBLS"]["series_params"] = api_params.get("USBLS").get("series_params")[:2]
    _ = get_data_from_apis(
        api_keys=api_keys,
        api_params=api_params,
        data_start_date=data_start_date,
        providers=["USBLS"],
    )


def test_get_data_oecd():
    """Test the whole get data workflow for the OECDGetter"""
    api_keys = API_MOCK_KEYS
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    api_params["OECD"]["series_params"] = [
        {
            "dataset_id": "QNA",
            "dimensions": {
                "countries": "FRA",
                "subject": "B1_GE",
                "measure": "GPSA",
                "frequency": "Q",
            },
        }
    ]
    _ = get_data_from_apis(
        api_keys=api_keys,
        api_params=api_params,
        data_start_date=data_start_date,
        providers=["OECD"],
    )


@pytest.mark.parametrize("api_keys", [{}, {"Mock_provider": "mock_key"}])
def test_get_data_key_error(api_keys):
    with pytest.raises(InvalidAPIKey):
        api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
        data_start_date = api_params.pop("data_start_date")
        api_params["FRED"]["series_params"] = api_params.get("FRED").get("series_params")[:2]
        _ = get_data_from_apis(
            api_keys=api_keys,
            api_params=api_params,
            data_start_date=data_start_date,
            providers=["FRED"],
        )


@pytest.mark.parametrize("api_params", [{}, {"Mock_provider": {"mock_params": "mock_value"}}])
def test_get_data_params_error(api_params):
    with pytest.raises(InvalidAPIRequestsParams):
        api_keys = API_MOCK_KEYS
        _ = get_data_from_apis(
            api_keys=api_keys,
            api_params=api_params,
            data_start_date="2010-08-01",
            providers=["FRED"],
        )
