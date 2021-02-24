from lib.data_retrieval.get_data import get_data_from_apis
from lib.utils.files import open_file
from config.config import API_REQUESTS_PARAMS_FILEPATH

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
    )


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
