import json
import pytest
import requests
from config.config import API_ENDPOINTS


@pytest.fixture(autouse=True)
def mock_response_get(monkeypatch):
    """requests.get mocked to return something based on MockAPIResponse class"""

    def mock_get(*args, **kwargs):
        return MockAPIResponse(*args, **kwargs)

    monkeypatch.setattr(requests, "get", mock_get)


class MockAPIResponse:
    """Produces a api mock response object for get requests"""

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs

    @staticmethod
    def raise_for_status():
        pass

    def json(self):
        """Return json formated response"""
        if self.url == API_ENDPOINTS["FRED_API_URL_SER"]:
            return self.fred_api_response_ser()
        if self.url == API_ENDPOINTS["FRED_API_URL_OBS"]:
            return self.fred_api_response_obs()
        if self.url == API_ENDPOINTS["USBLS_API_URL"]:
            return self.usbls_api_response()
        raise RuntimeError(
            "Network access not allowed during testing! "
            f"No mock response for this url: {self.url}"
        )

    def fred_api_response_obs(self):
        """ fred api response format for obesrvations"""
        params = self.kwargs.get("params")
        obs_val_dict = {
            "realtime_start": "2021-02-15",
            "realtime_end": "2021-02-15",
            "date": "1980-01-01",
            "value": "13.82",
        }
        response = {
            "realtime_start": "2021-02-15",
            "realtime_end": "2021-02-15",
            "observation_start": params.get("observation_start"),
            "observation_end": params.get("observation_end"),
            "units": params.get("units", "lin"),
            "output_type": 1,
            "file_type": params.get("file_type"),
            "order_by": "observation_date",
            "sort_order": "asc",
            "count": 3,
            "offset": 0,
            "limit": 100000,
            "observations": [obs_val_dict],
        }
        return response

    def fred_api_response_ser(self):
        """fred api response format for obesrvations"""
        params = self.kwargs.get("params")
        response = {
            "realtime_start": "2021-02-15",
            "realtime_end": "2021-02-15",
            "seriess": [
                {
                    "id": "FEDFUNDS",
                    "realtime_start": "2021-02-15",
                    "realtime_end": "2021-02-15",
                    "title": "Mock series name",
                    "observation_start": params.get("observation_start"),
                    "observation_end": params.get("observation_end"),
                    "notes": "mock notes",
                }
            ],
        }
        return response

    def usbls_api_response(self, seriesids):
        raise NotImplementedError


def load_test_data(filepath):
    """Load test data from json files
    """
    with open(filepath) as data:
        my_dict = json.load(data)
    return my_dict


@pytest.fixture
def expected_result_get_fred_data():
    data = tuple(load_test_data("tests/data_retrieval/expected_result_get_fred_data.json"))
    return data
