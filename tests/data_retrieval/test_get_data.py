import pytest

from lib.data_retrieval.get_data import get_data_from_apis
from lib.utils.files import open_file
from config.config import API_ENDPOINTS, API_KEYS_FILEPATH, API_REQUESTS_PARAMS_FILEPATH

def test_get_data_FRED():
    api_keys = open_file(API_KEYS_FILEPATH)
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    api_params["FRED"]["series_params"] = api_params.get("FRED").get("series_params")[:2]
    try:
        _ = get_data_from_apis(
            api_keys=api_keys,
            api_params=api_params,
            data_start_date=data_start_date,
            providers=["FRED"]
        )
    except Exception as exception:
        pytest.fail(str(exception))

def test_get_data_USBLS():
    api_keys = open_file(API_KEYS_FILEPATH)
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    api_params["USBLS"]["series_params"] = api_params.get("USBLS").get("series_params")[:2]
    try:
        _ = get_data_from_apis(
            api_keys=api_keys,
            api_params=api_params,
            data_start_date=data_start_date,
            providers=["USBLS"]
        )
    except Exception as exception:
        pytest.fail(str(exception))


def test_get_data_OECD():
    api_keys = open_file(API_KEYS_FILEPATH)
    api_params = open_file(API_REQUESTS_PARAMS_FILEPATH)
    data_start_date = api_params.pop("data_start_date")
    api_params["OECD"]["series_params"] = [{"dataset_id": "QNA", "dimensions": {"countries": "FRA", "subject": "B1_GE", "measure": "GPSA", "frequency": "Q"}}]
    try:
        _ = get_data_from_apis(
            api_keys=api_keys,
            api_params=api_params,
            data_start_date=data_start_date,
            providers=["OECD"]
        )
    except Exception as exception:
        pytest.fail(str(exception))
    
